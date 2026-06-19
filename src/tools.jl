#########################################################
# Prealloc
#########################################################
mutable struct PreAllocated{M, C}
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
$(TYPEDSIGNATURES)

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
    I4,     # <:ConvolutionPlan or EmptyConvolution,
    FP,     # <:ForwardPlan,
    IP,     # <:InversePlan,
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

function GIATools(domain, c, solidearth;
    quad_precision::Int = 4,
    rhs_smooth_radius = nothing)

    T = eltype(domain.R)

    viscous_green = domain.arraykernel(T.(
        green_viscous(
            domain,
            solidearth.rho_uppermantle,
            mean(solidearth.litho_rigidity),
        )))
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

    # FFT plans depending on CPU vs. GPU usage and mantle type
    pfft!, pifft! = choose_fft_plans(domain.K, solidearth.mantle)

    n_cplx_matrices = 3
    realmatrices = [kernelzeros(domain) for _ in
        eachindex(fieldnames(PreAllocated))[1:end-n_cplx_matrices]]
    cplxmatrices = _make_cplx_matrices(domain, solidearth.mantle, n_cplx_matrices)
    prealloc = PreAllocated(realmatrices..., cplxmatrices...)
    return GIATools(conv_helpers, viscous_convo, elastic_convo, dz_ss_convo,
        smooth_convo, pfft!, pifft!, prealloc)
end


function choose_fft_plans(X)
    return plan_fft!(complex.(X); flags = MEASURE), plan_ifft!(complex.(X); flags = MEASURE)
end

function choose_fft_plans(X, mantle)
    if mantle isa RealMaxwellMantle && X isa AbstractMatrix
        @warn "RealMaxwellMantle is experimental: it may yield larger numerical errors " *
              "than MaxwellMantle for laterally-variable lithosphere setups, and the " *
              "expected performance gain may not materialise on all hardware. " *
              "Prefer MaxwellMantle for production runs."
        rfft_buf = similar(X, Complex{eltype(X)}, size(X, 1) ÷ 2 + 1, size(X, 2))
        return plan_rfft(copy(X); flags = MEASURE), plan_irfft(rfft_buf, size(X, 1); flags = MEASURE)
    else
        return choose_fft_plans(X)
    end
end

function _make_cplx_matrices(domain, mantle, n)
    if mantle isa RealMaxwellMantle
        T = eltype(domain.R)
        nx2 = domain.nx ÷ 2 + 1
        return [domain.arraykernel(zeros(Complex{T}, nx2, domain.ny)) for _ in 1:n]
    else
        return [complex.(kernelzeros(domain)) for _ in 1:n]
    end
end