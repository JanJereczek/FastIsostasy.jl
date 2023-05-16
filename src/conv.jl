using FFTW
using DSP

struct PlannedConv
    Bpad::Matrix
    fftA::Matrix
    pfft::AbstractFFTs.Plan
    pifft::AbstractFFTs.ScaledPlan
    N::Int
    M1::Int
    M2::Int
end

function PlannedConv(A::Matrix, B::Matrix)
    if size(A) != size(B)
        error("Planned convolution only implemented for matrices of same size.")
    elseif size(A, 1) != size(A, 2)
        error("Planned convolution only implemented for square matrices.")
    else
        N = size(A, 1)
        Apad = zeros(eltype(A), 2*N-1, 2*N-1)
        view(Apad, 1:N, 1:N) .= A
        pfft = plan_fft(Apad)
        pifft = plan_ifft(Apad)
        return PlannedConv( copy(Apad), pfft * Apad, pfft, pifft, N, N ÷ 2, N + N ÷ 2 - 1)
    end
end

function planned_conv(B, p)
    view(p.Bpad, 1:p.N, 1:p.N) .= B
    return real.( view((p.pifft * ( p.fftA .* (p.pfft * p.Bpad) )), p.M1:p.M2, p.M1:p.M2) )
end

N = 100
M1 = N ÷ 2
M2 = N + N ÷ 2 - 1
A, B = rand(N, N), rand(N, N)
@btime C = DSP.conv($A, $B)[M1:M2, M1:M2]
p = PlannedConv(A, B)
@btime Cp = planned_conv($B, $p)

# D, E = fill(0.0, 2*N-1, 2*N-1), fill(0.0, 2*N-1, 2*N-1)
# D[1:N, 1:N] .= A
# E[1:N, 1:N] .= B
# F = ifft(fft(D) .* fft(E))[5:14, 5:14]


# a, b = rand(N), rand(N)
# c = conv(a, b)
# a0 = vcat(a, zeros(N-1))
# b0 = vcat(b, zeros(N-1))
# cp = ifft( fft( a0 ) .* fft(b0) )
