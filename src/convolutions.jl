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

    padded
end

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

const FAST_FFT_SIZES = (2, 3, 5, 7)
nextfastfft(n::Integer) = nextprod(FAST_FFT_SIZES, n)
nextfastfft(ns::Tuple{Vararg{Integer}}) = nextfastfft.(ns)

struct FastConvPlan{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    C<:ComplexMatrix{T},
    FP<:rFFTWPlan,
    IP<:AbstractFFTs.ScaledPlan}

    p_rfft::FP
    p_irfft::IP
    nffts::Tuple{Int64, Int64}
    kernel_padded::M
    input_padded::M
    kernel_fft::C
    input_fft::C
end

function FastConvPlan(kernel::KernelMatrix{T}) where T
    nx, ny = size(kernel)
    outsize = (2*nx-1, 2*ny-1)
    nffts = nextfastfft(outsize)
    kernel_padded = _zeropad!(similar(kernel, T, nffts), kernel)
    input_padded = similar(kernel_padded)
    p_rfft = plan_rfft(kernel_padded)
    kernel_fft = p_rfft * kernel_padded
    input_fft = similar(kernel_fft)
    p_irfft = plan_irfft(kernel_fft, nffts[1])
    return FastConvPlan(p_rfft, p_irfft, nffts, kernel_padded, input_padded, kernel_fft, input_fft)
end

function convo!(out, v, p::FastConvPlan)
    _zeropad!(p.input_padded, v)
    mul!(p.input_fft, p.p_rfft, p.input_padded)
    p.input_fft .*= p.kernel_fft
    mul!(out, p.p_irfft, p.input_fft)
    return nothing
end

struct InplaceConvolution{T<:AbstractFloat, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    pfft!::FP
    pifft!::IP
    kernel::M
    kernel_fft::C
    out_fft::C
    out::M
    buffer::M
    nx::Int
    ny::Int
    filler::T
end

function InplaceConvolution(kernel::M, use_cuda::Bool; filler::T = T(0)) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}

    nx, ny = size(kernel)
    kernel_fft = zeros(Complex{T}, 2*nx-1, 2*ny-1)
    buffer = zeros(T, 2*nx-1, 2*ny-1)
    view(kernel_fft, 1:nx, 1:ny) .= kernel
    if use_cuda
        kernel_fft = CuMatrix(kernel_fft)
        buffer = CuMatrix(buffer)
    end
    pfft!, pifft! = choose_fft_plans(kernel_fft, use_cuda)
    pfft! * kernel_fft
    return InplaceConvolution(pfft!, pifft!, kernel, kernel_fft, copy(kernel_fft),
        real.(kernel_fft), buffer, nx, ny, filler)
end

function convolution!(ipconv::I, B::M) where {I<:InplaceConvolution, M<:KernelMatrix}

    # (; pfft!, pifft!, kernel_fft, out_fft, out, nx, ny) = ipconv
    # out_fft .= ipconv.filler #background_value
    # view(out_fft, 1:nx, 1:ny) .= B
    # pfft! * out_fft
    # out_fft .*= kernel_fft
    # pifft! * out_fft
    # @. out = real(out_fft)

    conv!(out, ipconv.kernel, B)
    return nothing
end

"""
    samesize_conv(X, ipc, Omega)

Perform convolution of `X` with `ipc` and crop the result to the same size as `X`.
"""
function samesize_conv!(Y::M, X::M, ipc::InplaceConvolution{T, M, C, FP, IP},
    Omega::RegionalComputationDomain{T, L, M}, bc::OffsetBC, bc_space::ExtendedBCSpace) where {
        T<:AbstractFloat,
        L<:Matrix{T},
        M<:KernelMatrix{T},
        C<:ComplexMatrix{T},
        FP<:ForwardPlan{T},
        IP<:InversePlan{T}}
    
    convolution!(ipc, X)
    apply_bc!(ipc.out, bc)
    Y .= view(ipc.out,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
    return nothing
end

function samesize_conv!(Y::M, X::M, ipc::InplaceConvolution{T, M, C, FP, IP},
    Omega::RegionalComputationDomain{T, L, M}, bc::OffsetBC, bc_space::RegularBCSpace) where {
        T<:AbstractFloat,
        L<:Matrix{T},
        M<:KernelMatrix{T},
        C<:ComplexMatrix{T},
        FP<:ForwardPlan{T},
        IP<:InversePlan{T}}
    
    convolution!(ipc, X)
    Y .= view(ipc.out,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
    apply_bc!(Y, bc)
    return nothing
end

# Just a helper for blur! Not performant but we only blur at preprocessing
# so we do not care :)
function samesize_conv(X::L, Y::M, Omega::RegionalComputationDomain, filler::T;
    on_gpu = false) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    (; i1, i2, j1, j2, convo_offset) = Omega
    ipc = InplaceConvolution(X, on_gpu; filler=filler)
    return samesize_conv(Y, ipc, i1, i2, j1, j2, convo_offset)
end
function samesize_conv(Y, ipc::InplaceConvolution, i1, i2, j1, j2, convo_offset)
    convolution!(ipc, Y)
    return view(ipc.out, i1+convo_offset:i2+convo_offset,
        j1-convo_offset:j2-convo_offset)
end

function blur(X::AbstractMatrix, Omega::RegionalComputationDomain, level::Real, filler;
    on_gpu::Bool = false)
    if not(0 <= level <= 1)
        error("Blurring level must be a value between 0 and 1.")
    end
    T = eltype(X)
    sigma = T.(diagm([(level * Omega.Wx)^2, (level * Omega.Wy)^2]))
    kernel = generate_gaussian_field(Omega, T(0.0), T.([0.0, 0.0]), T(1.0), sigma)
    kernel ./= sum(kernel)
    # return copy(samesize_conv(Omega.arraykernel(kernel), Omega.arraykernel(X), Omega))
    return samesize_conv(kernel, X, Omega, filler; on_gpu = on_gpu)
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