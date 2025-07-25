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
    fftF::C
    fftU::C
end

#########################################################
# Tools
#########################################################
"""
    GIATools(domain, c, p)
Return a `struct` containing pre-computed tools to perform forward-stepping of the model.
This includes the Green's functions for the computation of the lithosphere and the SSH
perturbation, plans for FFTs, interpolators of the load and the viscosity over time and
preallocated arrays.
"""
struct GIATools{
    CPH<:ConvolutionPlanHelpers,
    I1<:ConvolutionPlan,
    I2<:ConvolutionPlan,
    I3<:ConvolutionPlan,
    I4, # <:ConvolutionPlan or EmptyConvolution,
    FP<:ForwardPlan,
    IP<:InversePlan,
    PA<:PreAllocated,
}
    conv_helpers::CPH
    viscous_convo::I1
    elastic_convo::I2
    dz_ss_convo::I3
    smooth_convo::I4
    pfft!::FP
    pifft!::IP
    prealloc::PA
end

function GIATools(domain, c, p; quad_precision::Int = 4, rhs_smooth_radius = nothing)

    T = eltype(domain.R)

    # Build in-place convolution for viscous response (only used in ELRA)
    L_w = get_flexural_lengthscale(mean(p.litho_rigidity), p.rho_uppermantle, c.g)
    kei = get_kei(domain, L_w)
    viscous_green = domain.arraykernel(T.(
        calc_viscous_green(domain, mean(p.litho_rigidity), kei, L_w)))
    conv_helpers = ConvolutionPlanHelpers(viscous_green)
    viscous_convo = ConvolutionPlan(viscous_green, conv_helpers)

    # Build in-place convolution to compute elastic response
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elastic_green = domain.arraykernel(T.(
        get_elastic_green(domain, greenintegrand_function, quad_support, quad_coeffs)))

    elastic_convo = ConvolutionPlan(elastic_green, conv_helpers)

    # Build in-place convolution to compute dz_ss response
    dz_ss_green = domain.arraykernel(T.(get_dz_ss_green(domain, c)))
    dz_ss_convo = ConvolutionPlan(dz_ss_green, conv_helpers)

    # Build in-place convolution for smoothing
    if isnothing(rhs_smooth_radius)
        smooth_convo = EmptyConvolution()
    else
        sigma = T.(diagm([(rhs_smooth_radius)^2, (rhs_smooth_radius)^2]))
        smoothing_kernel = generate_gaussian_field(domain, T(0.0), T.([0.0, 0.0]), T(1.0), sigma)
        norm!(smoothing_kernel)
        smooth_convo = ConvolutionPlan(domain.arraykernel(smoothing_kernel), conv_helpers)
    end

    # FFT plans depening on CPU vs. GPU usage
    pfft!, pifft! = choose_fft_plans(domain.K, domain.use_cuda)

    n_cplx_matrices = 3
    realmatrices = [kernelnull(domain) for _ in 
        eachindex(fieldnames(PreAllocated))[1:end-n_cplx_matrices]]
    cplxmatrices = [complex.(kernelnull(domain)) for _ in 1:n_cplx_matrices]
    prealloc = PreAllocated(realmatrices..., cplxmatrices...)
    return GIATools(conv_helpers, viscous_convo, elastic_convo, dz_ss_convo,
        smooth_convo, pfft!, pifft!, prealloc)
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