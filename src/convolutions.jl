"""
    _zeropad(u, padded_size, [data_dest, data_region])

Creates and returns a new base-1 index array of size `padded_size`, with the
section of `u` specified by `data_region` copied into the region of the new
 array as specified by `data_dest`. All other values will be initialized to
 zero.

If either `data_dest` or `data_region` is not specified, then the defaults
described in [`_zeropad!`](@ref) will be used.
"""
function _zeropad(u, padded_size, args...)
    padded = similar(u, padded_size)
    _zeropad!(padded, u, axes(padded), args...)
end

"""
    _zeropad!(padded, u, [padded_axes, data_dest, data_region])

Same as [`_zeropad`](@ref) but in place.
"""
@inline function _zeropad!(
    padded::AbstractArray,
    u::AbstractArray,
    padded_axes = axes(padded),
    data_dest::Tuple = first.(padded_axes),
    data_region = CartesianIndices(u),
)
    fill!(padded, zero(eltype(padded)))
    dest_axes = UnitRange.(data_dest, data_dest .+ size(data_region) .- 1)
    dest_region = CartesianIndices(dest_axes)
    copyto!(padded, dest_region, u, data_region)

    return nothing
end

const FAST_FFT_SIZES = (2, 3, 5, 7)
nextfastfft(n::Integer) = nextprod(FAST_FFT_SIZES, n)
nextfastfft(ns::Tuple{Vararg{Integer}}) = nextfastfft.(ns)

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
struct ConvolutionPlan{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    C<:ComplexMatrix{T},
    FP<:rFFTWPlan,
    IP<:AbstractFFTs.ScaledPlan}

    nx::Int
    ny::Int
    p_rfft::FP
    p_irfft::IP
    nffts::Tuple{Int64, Int64}
    kernel_padded::M
    input_padded::M
    output_padded::M
    output_cropped::M
    kernel_fft::C
    input_fft::C
end

function ConvolutionPlan(kernel::KernelMatrix{T}) where T
    nx, ny = size(kernel)
    outsize = (2*nx-1, 2*ny-1)
    nffts = nextfastfft(outsize)
    kernel_padded = similar(kernel, T, nffts)
    _zeropad!(kernel_padded, kernel)
    input_padded = similar(kernel_padded)
    output_padded = similar(kernel_padded)
    output_cropped = zeros(T, outsize)
    p_rfft = plan_rfft(kernel_padded)
    kernel_fft = p_rfft * kernel_padded
    input_fft = similar(kernel_fft)
    p_irfft = plan_irfft(kernel_fft, nffts[1])
    return ConvolutionPlan(nx, ny, p_rfft, p_irfft, nffts, kernel_padded, input_padded,
        output_padded, output_cropped, kernel_fft, input_fft)
end

function conv!(output, input, p::ConvolutionPlan)
    _zeropad!(p.input_padded, input)
    mul!(p.input_fft, p.p_rfft, p.input_padded)
    p.input_fft .*= p.kernel_fft
    mul!(output, p.p_irfft, p.input_fft)
    return nothing
end

function conv!(input, p::ConvolutionPlan)
    conv!(p.output_padded, input, p)
    p.output_cropped .= view(p.output_padded, 1:2*p.nx-1, 1:2*p.ny-1)
end

function conv(kernel, input)
    convplan = ConvolutionPlan(kernel)
    conv!(input, convplan)
    return convplan.output_padded
end

"""
    samesize_conv!(output, input, convplan, Omega, bc, bcspace)
"""
function samesize_conv!(output::M, input::M, p::ConvolutionPlan, Omega, bc,
    bc_space::ExtendedBCSpace) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    
    conv!(input, p)
    apply_bc!(p.output_cropped, bc)
    output .= view(p.output_cropped,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
    return nothing
end

function samesize_conv!(output::M, input::M, p::ConvolutionPlan, Omega, bc,
    bc_space::RegularBCSpace) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    
    conv!(input, p)
    output .= view(p.output_cropped,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
    apply_bc!(p.output_cropped, bc)
    return nothing
end

# TODO get rid of this!
# Just a helper for blur! Not performant but we only blur at preprocessing
# so we do not care :)
function samesize_conv(kernel, input, Omega::RegionalComputationDomain)
    (; i1, i2, j1, j2, convo_offset) = Omega
    p = convplan(kernel)
    return samesize_conv(input, p, i1, i2, j1, j2, convo_offset)
end
function samesize_conv(input, p::ConvolutionPlan, i1, i2, j1, j2, convo_offset)
    conv!(input, p)
    return p.output_cropped[i1+convo_offset:i2+convo_offset, j1-convo_offset:j2-convo_offset]
end

function blur(input, Omega::RegionalComputationDomain, level)
    if not(0 <= level <= 1)
        error("Blurring level must be a value between 0 and 1.")
    end
    T = eltype(input)
    sigma = T.(diagm([(level * Omega.Wx)^2, (level * Omega.Wy)^2]))
    kernel = generate_gaussian_field(Omega, T(0.0), T.([0.0, 0.0]), T(1.0), sigma)
    kernel ./= sum(kernel)
    return samesize_conv(kernel, input, Omega)
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