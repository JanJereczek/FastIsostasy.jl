@inline function _pad!(
    padded::AbstractArray,
    u::AbstractArray,
    pad_val::Real = 0,
    padded_axes = axes(padded),
    data_dest::Tuple = first.(padded_axes),
    data_region = CartesianIndices(u),
)
    fill!(padded, eltype(padded)(pad_val))
    dest_axes = UnitRange.(data_dest, data_dest .+ size(data_region) .- 1)
    dest_region = CartesianIndices(dest_axes)
    copyto!(padded, dest_region, u, data_region)

    return nothing
end

const FAST_FFT_SIZES = (2, 3, 5, 7)
nextfastfft(n::Integer) = nextprod(FAST_FFT_SIZES, n)
nextfastfft(ns::Tuple{Vararg{Integer}}) = nextfastfft.(ns)

"""
$(TYPEDSIGNATURES)

An unnormalized inverse FFT plan (the `bfft`/`brfft` inside a `ScaledPlan`) paired
with the exact normalization `S` that the `ScaledPlan` (`ifft`/`irfft`) would apply.
`mul!(y, np, x)` reproduces the `ScaledPlan` result bit-for-bit: `(raw_inverse * x) * S`.

`S` is carried as a TYPE PARAMETER rather than a field, so that it is a
compile-time constant and hence invisible to Enzyme. As a `Float64` field it would
be treated as differentiable whenever the plan lives inside an autodiff'd
[`Simulation`](@ref), corrupting gradients — the plan's scale is a constant, not a
differentiable quantity.
"""
struct NormalizedPlan{P,S}
    p::P
end
NormalizedPlan(p, scale::Real) = NormalizedPlan{typeof(p),Float64(scale)}(p)

"""
$(TYPEDSIGNATURES)

Replace a `ScaledPlan` (from `plan_ifft`/`plan_irfft`) with the equivalent
`NormalizedPlan`, extracting its raw plan and exact scale (build-extract-discard).
"""
normalize_plan(sp::AbstractFFTs.ScaledPlan) = NormalizedPlan(sp.p, sp.scale)
normalize_plan(p) = p

function LinearAlgebra.mul!(
    y::AbstractArray,
    np::NormalizedPlan{P,S},
    x::AbstractArray,
) where {P,S}
    LinearAlgebra.mul!(y, np.p, x)
    y .*= S
    return y
end

# CPU arrays get FFTW.MEASURE; GPU arrays fall through to the default (no-flag) method.
_plan_rfft(X::Matrix) = plan_rfft(X; flags = MEASURE)
_plan_rfft(X::AbstractMatrix) = plan_rfft(X)
_plan_irfft(X::Matrix, n::Int) = normalize_plan(plan_irfft(X, n; flags = MEASURE))
_plan_irfft(X::AbstractMatrix, n::Int) = normalize_plan(plan_irfft(X, n))

"""
$(TYPEDSIGNATURES)

A helper for convolution plans.

# Fields
- `nx`: number of rows in the kernel
- `ny`: number of columns in the kernel
- `p_rfft`: the real-valued FFT plan
- `p_irfft`: the real-valued inverse FFT plan (including scaling)
- `nffts`: the padded size of the FFTs
- `kernel_padded`: the padded kernel to convolve the input with
- `input_padded`: the padded input
- `output_padded`: the padded output
- `output_cropped`: the cropped output
- `input_fft`: the transformed (padded) input
- `pad_val`: the value used to pad the input and kernel
"""
struct ConvolutionPlanHelpers{T,M,C,FP,IP}
    nx::Int
    ny::Int
    p_rfft::FP
    p_irfft::IP
    nffts::Tuple{Int64,Int64}
    kernel_padded::M
    input_padded::M
    output_padded::M
    output_cropped::M
    input_fft::C
    pad_val::T
end

function ConvolutionPlanHelpers(kernel::AbstractMatrix; pad_val = 0)
    nx, ny = size(kernel)
    T = eltype(kernel)
    outsize = (2*nx-1, 2*ny-1)
    nffts = nextfastfft(outsize)
    kernel_padded = similar(kernel, T, nffts)
    _pad!(kernel_padded, kernel, 0)
    input_padded = similar(kernel_padded)
    output_padded = similar(kernel_padded)
    output_cropped = similar(kernel, outsize...)
    p_rfft = _plan_rfft(kernel_padded)
    kernel_fft = p_rfft * kernel_padded
    input_fft = similar(kernel_fft)
    p_irfft = _plan_irfft(kernel_fft, nffts[1])
    return ConvolutionPlanHelpers(
        nx,
        ny,
        p_rfft,
        p_irfft,
        nffts,
        kernel_padded,
        input_padded,
        output_padded,
        output_cropped,
        input_fft,
        T(pad_val),
    )
end

"""
$(TYPEDSIGNATURES)

A convolution plan that precomputes the FFT of a kernel for repeated convolutions.
Typically used in combination with a [`ConvolutionPlanHelpers`](@ref):

```julia
helpers = ConvolutionPlanHelpers(kernel)
plan = ConvolutionPlan(kernel, helpers)
conv!(output, input, plan, helpers)
```

If you want the output to be the same size as the input, use [`samesize_conv`](@ref)
instead of `conv!`. The `samesize_conv` function will automatically crop the output
to the same size as the input, and apply boundary conditions if provided.

# Fields
- `kernel`: the kernel to convolve the input with
- `kernel_fft`: the transformed (padded) kernel
"""
struct ConvolutionPlan{M,C}
    kernel::M
    kernel_fft::C
end

function ConvolutionPlan(kernel::AbstractMatrix, helpers::ConvolutionPlanHelpers)
    _pad!(helpers.kernel_padded, kernel, 0)
    return ConvolutionPlan(kernel, helpers.p_rfft * helpers.kernel_padded)
end

"""
$(TYPEDSIGNATURES)

Convolve `input` with the kernel stored in `p`, using the helper `h` to store
intermediate results. The output is stored in `output`, which must be the same
size as `h.output_padded`.
"""
function conv!(output, input, p::ConvolutionPlan, h::ConvolutionPlanHelpers)
    _pad!(h.input_padded, input, h.pad_val)
    mul!(h.input_fft, h.p_rfft, h.input_padded)
    h.input_fft .*= p.kernel_fft
    mul!(output, h.p_irfft, h.input_fft)
    return nothing
end

function conv!(input, p::ConvolutionPlan, h::ConvolutionPlanHelpers)
    conv!(h.output_padded, input, p, h)
    h.output_cropped .= view(h.output_padded, 1:(2*h.nx-1), 1:(2*h.ny-1))
    return nothing
end

function conv(kernel, input)
    convhelpers = ConvolutionPlanHelpers(kernel)
    convplan = ConvolutionPlan(kernel, convhelpers)
    conv!(input, convplan, convhelpers)
    return convhelpers.output_padded
end

struct EmptyConvolution end

"""
$(TYPEDSIGNATURES)

Convolve `input` with the kernel stored in `p`, using the helper `h` to store
intermediate results.
"""
function samesize_conv!(output, input, p::EmptyConvolution, h, domain)
    return nothing
end

# Crop the (2n-1)-sized convolution result back onto the computation grid.
function crop_conv!(output, h::ConvolutionPlanHelpers, domain)
    output .= view(
        h.output_cropped,
        (domain.i1+domain.convo_offset):(domain.i2+domain.convo_offset),
        (domain.j1-domain.convo_offset):(domain.j2-domain.convo_offset),
    )
    return nothing
end

function samesize_conv!(
    output::M,
    input::M,
    p::ConvolutionPlan,
    h::ConvolutionPlanHelpers,
    domain,
) where {M}

    conv!(input, p, h)
    crop_conv!(output, h, domain)
    return nothing
end

# The two BC-carrying methods differ only in *when* the BC is applied: on the
# extended grid the convolution produced (before cropping), or on the computation
# grid (after). That ordering is the whole point of `AbstractBCSpace`, so they
# cannot collapse into one.
function samesize_conv!(
    output::M,
    input::M,
    p::ConvolutionPlan,
    h::ConvolutionPlanHelpers,
    domain,
    bc,
    bc_space::ExtendedBCSpace,
) where {M}

    conv!(input, p, h)
    apply_bc!(h.output_cropped, bc)
    crop_conv!(output, h, domain)
    return nothing
end

function samesize_conv!(
    output::M,
    input::M,
    p::ConvolutionPlan,
    h::ConvolutionPlanHelpers,
    domain,
    bc,
    bc_space::RegularBCSpace,
) where {M}

    conv!(input, p, h)
    crop_conv!(output, h, domain)
    apply_bc!(output, bc)
    return nothing
end

function samesize_conv(kernel, input, domain::RegionalDomain; pad_val = 0)
    (; i1, i2, j1, j2, convo_offset, backend) = domain
    h = ConvolutionPlanHelpers(kernel; pad_val = pad_val)
    p = ConvolutionPlan(kernel, h)
    return samesize_conv(input, p, h, i1, i2, j1, j2, convo_offset, backend)
end
function samesize_conv(
    input,
    p::ConvolutionPlan,
    h::ConvolutionPlanHelpers,
    i1,
    i2,
    j1,
    j2,
    convo_offset,
    backend,
)
    conv!(input, p, h)
    return kernelpromote(
        h.output_cropped[
            (i1+convo_offset):(i2+convo_offset),
            (j1-convo_offset):(j2-convo_offset),
        ],
        backend,
    )
end

function gaussian_smooth(
    input,
    domain::RegionalDomain,
    level::R,
    pad_val,
) where {R<:Real}

    if not(0 <= level <= 1)
        error("Blurring level must be a value between 0 and 1.")
    end
    T = eltype(input)
    sigma = T.(diagm([(level * domain.Wx)^2, (level * domain.Wy)^2]))
    kernel = generate_gaussian_field(domain, T(0.0), T.([0.0, 0.0]), T(1.0), sigma)
    kernel ./= sum(kernel)
    return samesize_conv(kernel, input, domain; pad_val = pad_val)
end

function gaussian_smooth(input, X, Y, s_x, s_y)
    T = eltype(input)
    σ = T.(diagm([(s_x)^2, (s_y)^2]))
    kernel = gauss_distr(X, Y, T.([0, 0]), σ)
    kernel ./= sum(kernel)
    return conv(kernel, input)
end


"""
$(TYPEDSIGNATURES)

Get the start and end indices required for a [`samesize_conv`](@ref)
"""
function samesize_conv_indices(N, M)
    if iseven(N)
        j1 = M
    else
        j1 = M+1
    end
    j2 = 2*N-1-M
    return j1, j2
end