struct InplaceConvolution{T<:AbstractFloat, C<:ComplexMatrix{T}, FP<:ForwardPlan{T},
    IP<:InversePlan{T}}
    pfft!::FP
    pifft!::IP
    Afft::C
    Bfft::C
    Cfft::C
    Nx::Int
    Ny::Int
end

function InplaceConvolution(A::M, use_cuda::Bool) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    Nx, Ny = size(A)
    if use_cuda
        Afft = CuMatrix(zeros(Complex{T}, 2*Nx-1, 2*Ny-1))
    else
        Afft = zeros(Complex{T}, 2*Nx-1, 2*Ny-1)
    end
    view(Afft, 1:Nx, 1:Ny) .= A
    pfft!, pifft! = choose_fft_plans(Afft, use_cuda)
    pfft! * Afft
    return InplaceConvolution(pfft!, pifft!, Afft, copy(Afft), copy(Afft), Nx, Ny)
end

function (ipconv::InplaceConvolution{T, C, FP, IP})(B::M) where {T<:AbstractFloat,
    M<:KernelMatrix{T}, C<:ComplexMatrix{T}, FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    (; pfft!, pifft!, Afft, Bfft, Cfft, Nx, Ny) = ipconv
    Bfft .= 0
    view(Bfft, 1:Nx, 1:Ny) .= complex.(B)
    pfft! * Bfft
    Cfft .= Afft .* Bfft
    pifft! * Cfft
    return real.(Cfft)
end