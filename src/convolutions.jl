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

struct ConvolutionPlanHelpers{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    C<:ComplexMatrix{T},
    FP,         # <:rFFTWPlan,
    IP,         #<:AbstractFFTs.ScaledPlan
}
    nx::Int
    ny::Int
    p_rfft::FP
    p_irfft::IP
    nffts::Tuple{Int64, Int64}
    kernel_padded::M
    input_padded::M
    output_padded::M
    output_cropped::M
    input_fft::C
    pad_val::T
end

function ConvolutionPlanHelpers(kernel::KernelMatrix{T}; pad_val = 0) where T
    nx, ny = size(kernel)
    outsize = (2*nx-1, 2*ny-1)
    nffts = nextfastfft(outsize)
    kernel_padded = similar(kernel, T, nffts)
    _pad!(kernel_padded, kernel, 0)
    input_padded = similar(kernel_padded)
    output_padded = similar(kernel_padded)
    output_cropped = similar(kernel, outsize...)
    p_rfft = plan_rfft(kernel_padded)
    kernel_fft = p_rfft * kernel_padded
    input_fft = similar(kernel_fft)
    p_irfft = plan_irfft(kernel_fft, nffts[1])
    return ConvolutionPlanHelpers(nx, ny, p_rfft, p_irfft, nffts, kernel_padded,
        input_padded, output_padded, output_cropped, input_fft, T(pad_val))
end

"""
    ConvolutionPlan

A struct that contains:
 - p_rfft: the real-valued FFT plan
 - p_irfft: the real-valued inverse FFT plan (including scaling)
 - nffts: 
 - kernel_padded: the padded kernel to convolve the input with
 - input_padded: the padded input
 - kernel_fft: the transformed (padded) kernel
 - input_fft: the transformed (padded) input

To initialize the plan and perform a convolution based on it:
```julia
kernel, input = [rand(64, 64) for _ in 1:2]
convplan = ConvolutionPlan(kernel)
conv!(input, convplan)
```

You can retrieve the result by calling `convplan.output_padded` but the
result will be on the padded domain. To retrieve the result on the original
domain, use [`samesize_conv!`](@ref).
"""
struct ConvolutionPlan{C<:ComplexMatrix}
    kernel_fft::C
end

function ConvolutionPlan(kernel::KernelMatrix{T}, helpers::ConvolutionPlanHelpers) where T
    _pad!(helpers.kernel_padded, kernel, 0)
    return ConvolutionPlan(helpers.p_rfft * helpers.kernel_padded)
end

function conv!(output, input, p::ConvolutionPlan, h::ConvolutionPlanHelpers)
    _pad!(h.input_padded, input, h.pad_val)
    mul!(h.input_fft, h.p_rfft, h.input_padded)
    h.input_fft .*= p.kernel_fft
    mul!(output, h.p_irfft, h.input_fft)
    return nothing
end

function conv!(input, p::ConvolutionPlan, h::ConvolutionPlanHelpers)
    conv!(h.output_padded, input, p, h)
    h.output_cropped .= view(h.output_padded, 1:2*h.nx-1, 1:2*h.ny-1)
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
    samesize_conv!(output, input, convplan, domain, bc, bcspace)
"""
function samesize_conv!(output, input, p::EmptyConvolution, h, domain)
    return nothing
end

function samesize_conv!(output::M, input::M, p::ConvolutionPlan,
    h::ConvolutionPlanHelpers, domain) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    
    conv!(input, p, h)
    output .= view(h.output_cropped,
        domain.i1+domain.convo_offset:domain.i2+domain.convo_offset,
        domain.j1-domain.convo_offset:domain.j2-domain.convo_offset)
    return nothing
end

function samesize_conv!(output::M, input::M, p::ConvolutionPlan, h::ConvolutionPlanHelpers,
    domain, bc, bc_space::ExtendedBCSpace) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    
    conv!(input, p, h)
    apply_bc!(h.output_cropped, bc)
    output .= view(h.output_cropped,
        domain.i1+domain.convo_offset:domain.i2+domain.convo_offset,
        domain.j1-domain.convo_offset:domain.j2-domain.convo_offset)
    return nothing
end

function samesize_conv!(output::M, input::M, p::ConvolutionPlan, h::ConvolutionPlanHelpers,
    domain, bc, bc_space::RegularBCSpace) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    
    conv!(input, p, h)
    output .= view(h.output_cropped,
        domain.i1+domain.convo_offset:domain.i2+domain.convo_offset,
        domain.j1-domain.convo_offset:domain.j2-domain.convo_offset)
    apply_bc!(output, bc)
    return nothing
end

function samesize_conv(kernel, input, domain::RegionalDomain; pad_val = 0)
    (; i1, i2, j1, j2, convo_offset, arraykernel) = domain
    h = ConvolutionPlanHelpers(kernel; pad_val = pad_val)
    p = ConvolutionPlan(kernel, h)
    return samesize_conv(input, p, h, i1, i2, j1, j2, convo_offset, arraykernel)
end
function samesize_conv(input, p::ConvolutionPlan, h::ConvolutionPlanHelpers,
    i1, i2, j1, j2, convo_offset, arraykernel)
    conv!(input, p, h)
    return arraykernel(h.output_cropped[i1+convo_offset:i2+convo_offset,
        j1-convo_offset:j2-convo_offset])
end

function gaussian_smooth(input, domain::RegionalDomain, level::R, pad_val) where
    {R<:Real}

    if not(0 <= level <= 1)
        error("Blurring level must be a value between 0 and 1.")
    end
    T = eltype(input)
    sigma = T.(diagm([(level * domain.Wx)^2, (level * domain.Wy)^2]))
    kernel = generate_gaussian_field(domain, T(0.0), T.([0.0, 0.0]), T(1.0), sigma)
    kernel ./= sum(kernel)
    return samesize_conv(kernel, input, domain; pad_val = pad_val)
end


"""
    samesize_conv_indices(N, M)

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