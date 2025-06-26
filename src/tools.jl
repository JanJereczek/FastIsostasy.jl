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
    I1<:InplaceConvolution,
    I2<:InplaceConvolution,
    I3<:InplaceConvolution,
    FP<:ForwardPlan,
    IP<:InversePlan,
    PA<:PreAllocated,
}
    viscous_convo::I1
    elastic_convo::I2
    dz_ss_convo::I3
    pfft!::FP
    pifft!::IP
    prealloc::PA
end

function FastIsoTools(Omega, c, p; quad_precision::Int = 4)

    T = eltype(Omega.R)

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

    n_cplx_matrices = 1
    realmatrices = [kernelnull(Omega) for _ in 
        eachindex(fieldnames(PreAllocated))[1:end-n_cplx_matrices]]
    cplxmatrices = [complex.(kernelnull(Omega)) for _ in 1:n_cplx_matrices]
    prealloc = PreAllocated(realmatrices..., cplxmatrices...)
    
    return FastIsoTools(viscous_convo, elastic_convo, dz_ss_convo, pfft!, pifft!, prealloc)
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