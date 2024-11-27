#########################################################
# Prealloc
#########################################################
mutable struct PreAllocated{T<:AbstractFloat, M<:KernelMatrix{T}, C<:ComplexMatrix{T}}
    rhs::M
    buffer_xx::M
    buffer_yy::M
    buffer_x::M
    buffer_xy::M
    Mxx::M
    Myy::M
    Mxy::M
    fftrhs::C
end

#########################################################
# Tools
#########################################################
"""
    FastIsoTools(Omega, c, p)
Return a `struct` containing pre-computed tools to perform forward-stepping of the model.
This includes the Green's functions for the computation of the lithosphere and the SSH
perturbation, plans for FFTs, interpolators of the load and the viscosity over time and
preallocated arrays.
"""
struct FastIsoTools{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    C<:ComplexMatrix{T},
    FP<:ForwardPlan{T},
    IP<:InversePlan{T},
}

    viscous_convo::InplaceConvolution{T, M, C, FP, IP}
    elastic_convo::InplaceConvolution{T, M, C, FP, IP}
    dz_ss_convo::InplaceConvolution{T, M, C, FP, IP}
    pfft!::FP
    pifft!::IP
    Hice::TimeInterpolation2D{T, M}
    bsl::Interpolations.Extrapolation{T, 1,
        Interpolations.GriddedInterpolation{T, 1, Vector{T},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}},
        Gridded{Linear{Throw{OnGrid}}}, Flat{Nothing}}
    prealloc::PreAllocated{T, M, C}
end

function FastIsoTools(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}},
    bsl_itp; quad_precision::Int = 4,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}

    # Build in-place convolution for viscous response (only used in ELRA)
    L_w = get_flexural_lengthscale(mean(p.litho_rigidity), p.rho_uppermantle, c.g)
    kei = get_kei(Omega, L_w)
    viscousgreen = calc_viscous_green(Omega, mean(p.litho_rigidity), kei, L_w)
    viscous_convo = InplaceConvolution(T.(viscousgreen), Omega.use_cuda)

    # Build in-place convolution to compute elastic response
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elasticgreen = get_elasticgreen(Omega, greenintegrand_function, quad_support, quad_coeffs)
    elastic_convo = InplaceConvolution(T.(elasticgreen), Omega.use_cuda)

    # Build in-place convolution to compute dz_ss response
    dz_ssgreen = get_dz_ssgreen(Omega, c)
    dz_ss_convo = InplaceConvolution(T.(dz_ssgreen), Omega.use_cuda)

    # FFT plans depening on CPU vs. GPU usage
    pfft!, pifft! = choose_fft_plans(Omega.K, Omega.use_cuda)

    Hice = TimeInterpolation2D(
        t_Hice_snapshots, kernelpromote(Hice_snapshots, Omega.arraykernel))
    n_cplx_matrices = 1
    realmatrices = [kernelnull(Omega) for _ in 
        eachindex(fieldnames(PreAllocated))[1:end-n_cplx_matrices]]
    cplxmatrices = [complex.(kernelnull(Omega)) for _ in 1:n_cplx_matrices]
    prealloc = PreAllocated(realmatrices..., cplxmatrices...)
    
    # @show typeof(Hice) typeof(bsl_itp)
    return FastIsoTools(viscous_convo, elastic_convo, dz_ss_convo, pfft!, pifft!,
        Hice, bsl_itp, prealloc)
end


function choose_fft_plans(X, use_cuda)
    if use_cuda
        pfft! = CUFFT.plan_fft!(complex.(X))
        pifft! = CUFFT.plan_ifft!(complex.(X))
    else
        pfft! = plan_fft!(complex.(X))
        pifft! = plan_ifft!(complex.(X))
    end
    return pfft!, pifft!
end
