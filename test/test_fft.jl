using FFTW, LinearAlgebra

N = 100
A = rand(N, N)
p = plan_fft(A)
A2 = complex.(zeros(N, N))
@btime p*A
@btime mul!(A2, p, A, 1.0, 1.0)


N = 100
A = complex.(rand(N, N))
p = plan_fft!(A)
p * A
@btime p*A
B = rand(N, N)
@btime $A .= complex.($B)

using DSP
N = 10
A = rand(N, N)
B = rand(N, N)
C = rand(N, N)

lhs = (conv(A, C .* B) + conv(B, C .* A)) ./ C
rhs = conv(A, B) + conv(B, A)