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