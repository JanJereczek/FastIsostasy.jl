#########################################################
# FFT backend
#########################################################

"""
$(TYPEDSIGNATURES)

Which FFT plans the spectral solver builds. This is a *numerical* choice and is
deliberately orthogonal to the rheology: it selects how a transform is computed,
never what is being modelled. Set it through the `fft` field of
[`SolverOptions`](@ref).

Available subtypes:
- [`ComplexFFTBackend`](@ref)
- [`RealFFTBackend`](@ref)
"""
abstract type AbstractFFTBackend end

"""
$(TYPEDSIGNATURES)

Complex-to-complex plans (`plan_fft` / `plan_ifft`) over the full `(nx, ny)`
spectrum. The default, and the FFT backend to use for production runs.
"""
struct ComplexFFTBackend <: AbstractFFTBackend end

"""
$(TYPEDSIGNATURES)

Real-to-complex plans (`plan_rfft` / `plan_irfft`). The frequency-domain arrays
are `(nx÷2+1, ny)` rather than `(nx, ny)`, roughly halving the memory and
arithmetic cost of the spectral step.

!!! warning "Experimental"
    In laterally-variable lithosphere setups the half-spectrum views introduce
    extra complexity that can produce larger numerical errors than
    [`ComplexFFTBackend`](@ref), and the expected performance gain may not
    materialise on all hardware. A runtime warning is emitted when this FFT
    backend is selected.
"""
struct RealFFTBackend <: AbstractFFTBackend end

#########################################################
# Prealloc
#########################################################
"""
$(TYPEDSIGNATURES)

Preallocated work arrays of the [`GIATools`](@ref), so that the right-hand side of
the model is allocation-free.

# Fields
$(TYPEDFIELDS)
"""
mutable struct PreAllocated{M,C,C3}
    "the right-hand side of the deformation equation"
    rhs::M
    "a real buffer"
    buffer_xx::M
    "a real buffer"
    buffer_yy::M
    "a real buffer, e.g. for reductions over the domain"
    buffer_x::M
    "a real buffer"
    buffer_xy::M
    "the bending moment `Mxx` of the thin plate"
    Mxx::M
    "the bending moment `Myy` of the thin plate"
    Myy::M
    "the bending moment `Mxy` of the thin plate"
    Mxy::M
    "the complex staging buffer of the forward FFTs"
    fftrhs::C
    "the FFT of the load term"
    fftF::C
    "the FFT of the viscous displacement"
    fftU::C
    """
    the FFTs of the Kelvin-branch displacements of a [`TransientCreepMantle`](@ref),
    stacked along dim 3 exactly like `u_K` in `CurrentState`; zero-sized
    otherwise
    """
    fftK::C3
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

# Fields
$(TYPEDFIELDS)
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
    "the `ConvolutionPlanHelpers` shared by all convolution plans"
    conv_helpers::CPH
    "the `ConvolutionPlan` with the viscous Green's function"
    viscous_convo::I1
    "the `ConvolutionPlan` with the elastic Green's function"
    elastic_convo::I2
    "the `ConvolutionPlan` with the Green's function of the sea-surface perturbation"
    dz_ss_convo::I3
    "the `ConvolutionPlan` smoothing the right-hand side, or `EmptyConvolution` if none"
    smooth_convo::I4
    "the forward FFT plan"
    pfft!::FP
    "the inverse FFT plan"
    pifft!::IP
    "the `PreAllocated` work arrays"
    prealloc::PA
end

function GIATools(
    domain,
    c,
    solidearth;
    quad_precision::Int = 4,
    rhs_smooth_radius = nothing,
    fft::AbstractFFTBackend = ComplexFFTBackend(),
)

    T = eltype(domain.R)

    viscous_green = kernelpromote(
        T.(
            green_viscous(
                domain,
                solidearth.rho_uppermantle,
                mean(solidearth.litho_rigidity),
                c.g,
            ),
        ),
        domain.backend,
    )
    conv_helpers = ConvolutionPlanHelpers(viscous_green)
    viscous_convo = ConvolutionPlan(viscous_green, conv_helpers)

    # Build in-place convolution to compute elastic response
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elastic_green = kernelpromote(
        T.(
            get_elastic_green(
                domain,
                greenintegrand_function,
                quad_support,
                quad_coeffs,
            ),
        ),
        domain.backend,
    )

    elastic_convo = ConvolutionPlan(elastic_green, conv_helpers)

    # Build in-place convolution to compute dz_ss response
    dz_ss_green = kernelpromote(T.(get_dz_ss_green(domain, c)), domain.backend)
    dz_ss_convo = ConvolutionPlan(dz_ss_green, conv_helpers)

    # Build in-place convolution for smoothing
    if isnothing(rhs_smooth_radius)
        smooth_convo = EmptyConvolution()
    else
        sigma = T.(diagm([(rhs_smooth_radius)^2, (rhs_smooth_radius)^2]))
        smoothing_kernel =
            generate_gaussian_field(domain, T(0.0), T.([0.0, 0.0]), T(1.0), sigma)
        norm!(smoothing_kernel)
        smooth_convo =
            ConvolutionPlan(kernelpromote(smoothing_kernel, domain.backend), conv_helpers)
    end

    # `domain.backend` decides host vs. device planning, `fft` complex vs. real
    pfft!, pifft! = choose_fft_plans(domain.K, fft)

    n_cplx_matrices = 4
    realmatrices = [
        kernelzeros(domain) for
        _ in eachindex(fieldnames(PreAllocated))[1:(end-n_cplx_matrices)]
    ]
    cplxmatrices = _make_cplx_matrices(domain, fft, n_cplx_matrices - 1)
    fftK = _make_cplx_branch_array(domain, fft, nbranches(solidearth.mantle))
    prealloc = PreAllocated(realmatrices..., cplxmatrices..., fftK)
    return GIATools(
        conv_helpers,
        viscous_convo,
        elastic_convo,
        dz_ss_convo,
        smooth_convo,
        pfft!,
        pifft!,
        prealloc,
    )
end


# Out-of-place complex plans: applied via `mul!(dest, plan, src)`, which preserves
# `src`. Input preservation is required for AD (the primal input to each transform
# must survive for the reverse pass) and keeps the forward code allocation-free. The
# inverse plan is wrapped by `normalize_plan` (→ `NormalizedPlan`): numerically
# identical to the `ScaledPlan` from `plan_ifft`, but carrying its scale as a type
# parameter so Enzyme doesn't treat the (constant) normalization as differentiable.
#
# `plan_fft` & friends are `AbstractFFTs` generics: FFTW claims them for host
# arrays, and every GPU package claims them for its own array type (CUFFT, rocFFT,
# …). So the planner needs *no* vendor-specific method — the one thing that is not
# portable is the FFTW planner-effort flag, which only the host planner accepts.
# Selecting that off the backend is what lets a new GPU vendor cost zero lines here.
_planner_flags(::CPU) = (; flags = MEASURE)
_planner_flags(::Backend) = (;)

function choose_fft_plans(X)
    kw = _planner_flags(get_backend(X))
    return plan_fft(complex.(X); kw...),
    normalize_plan(plan_ifft(complex.(X); kw...))
end

choose_fft_plans(X, ::ComplexFFTBackend) = choose_fft_plans(X)

# Half-spectrum plans. Falls back to the complex ones for non-matrix `X`, which
# has no `nx÷2+1` layout to exploit.
function choose_fft_plans(X, ::RealFFTBackend)
    X isa AbstractMatrix || return choose_fft_plans(X)
    @warn "RealFFTBackend is experimental: it may yield larger numerical errors " *
          "than ComplexFFTBackend for laterally-variable lithosphere setups, and " *
          "the expected performance gain may not materialise on all hardware. " *
          "Prefer ComplexFFTBackend for production runs."
    kw = _planner_flags(get_backend(X))
    rfft_buf = similar(X, Complex{eltype(X)}, size(X, 1) ÷ 2 + 1, size(X, 2))
    return plan_rfft(copy(X); kw...),
    normalize_plan(plan_irfft(rfft_buf, size(X, 1); kw...))
end

_make_cplx_matrices(domain, ::ComplexFFTBackend, n) =
    [complex.(kernelzeros(domain)) for _ = 1:n]

function _make_cplx_matrices(domain, ::RealFFTBackend, n)
    T = eltype(domain.R)
    nx2 = domain.nx ÷ 2 + 1
    return [kernelzeros(domain.backend, Complex{T}, nx2, domain.ny) for _ = 1:n]
end

_make_cplx_branch_array(domain, ::ComplexFFTBackend, N) =
    kernelzeros(domain.backend, Complex{eltype(domain.R)}, domain.nx, domain.ny, N)

function _make_cplx_branch_array(domain, ::RealFFTBackend, N)
    T = eltype(domain.R)
    nx2 = domain.nx ÷ 2 + 1
    return kernelzeros(domain.backend, Complex{T}, nx2, domain.ny, N)
end
