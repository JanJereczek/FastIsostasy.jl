struct InplaceConvolution{T<:AbstractFloat, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    pfft!::FP
    pifft!::IP
    kernel_fft::C
    out_fft::C
    out::M
    Nx::Int
    Ny::Int
end

function InplaceConvolution(kernel::M, use_cuda::Bool) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    Nx, Ny = size(kernel)
    kernel_fft = zeros(Complex{T}, 2*Nx-1, 2*Ny-1)
    view(kernel_fft, 1:Nx, 1:Ny) .= kernel
    if use_cuda
        kernel_fft = CuMatrix(kernel_fft)
    end
    pfft!, pifft! = choose_fft_plans(kernel_fft, use_cuda)
    pfft! * kernel_fft
    return InplaceConvolution(pfft!, pifft!, kernel_fft, copy(kernel_fft), real.(kernel_fft), Nx, Ny)
end

function (ipconv::InplaceConvolution{T, M, C, FP, IP})(B::M) where {T<:AbstractFloat,
    M<:KernelMatrix{T}, C<:ComplexMatrix{T}, FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    (; pfft!, pifft!, kernel_fft, out_fft, out, Nx, Ny) = ipconv
    out_fft .= 0
    view(out_fft, 1:Nx, 1:Ny) .= complex.(B)
    pfft! * out_fft
    out_fft .*= kernel_fft
    pifft! * out_fft
    @. out = real(out_fft)
    return nothing
end

"""
    samesize_conv(X, ipc, Omega)

Perform convolution of `X` with `ipc` and crop the result to the same size as `X`.
"""
function samesize_conv(X::M, ipc::InplaceConvolution{T, M, C, FP, IP},
    Omega::ComputationDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T},
    M<:KernelMatrix{T}, C<:ComplexMatrix{T}, FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    ipc(X)
    apply_bc!(ipc.out, Omega.extended_bc_matrix, Omega.extended_nbc)
    return view(ipc.out,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
end

# Just a helper for blur! Not performant but we only blur at preprocessing
# so we do not care :)
function samesize_conv(X::M, Y::M, Omega::ComputationDomain) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    (; i1, i2, j1, j2, convo_offset) = Omega
    ipc = InplaceConvolution(X, false)
    return samesize_conv(Y, ipc, i1, i2, j1, j2, convo_offset)
end
function samesize_conv(Y, ipc, i1, i2, j1, j2, convo_offset)
    ipc(Y)
    return view(ipc.out, i1+convo_offset:i2+convo_offset,
        j1-convo_offset:j2-convo_offset)
end

function blur(X::AbstractMatrix, Omega::ComputationDomain, level::Real)
    if not(0 <= level <= 1)
        error("Blurring level must be a value between 0 and 1.")
    end
    T = eltype(X)
    sigma = diagm([(level * Omega.Wx)^2, (level * Omega.Wy)^2])
    kernel = T.(generate_gaussian_field(Omega, 0.0, [0.0, 0.0], 1.0, sigma))
    kernel ./= sum(kernel)
    # return copy(samesize_conv(Omega.arraykernel(kernel), Omega.arraykernel(X), Omega))
    return samesize_conv(kernel, X, Omega)
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

