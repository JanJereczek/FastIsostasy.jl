function derivative_stdsetup(use_cuda::Bool)
    W, n = 3000e3, 7
    Omega = ComputationDomain(W, n, correct_distortion = false, use_cuda = use_cuda)
    c, p, t_out, _, _, t_Hice, Hice = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice)

    u = Omega.X .^ 2 .* Omega.Y .^ 2
    uxx = 2 .* Omega.Y .^ 2
    uyy = 2 .* Omega.X .^ 2
    uxy = 4 .* Omega.X .* Omega.Y
    return Omega, fip.tools.prealloc, Omega.arraykernel(u), uxx, uyy, uxy
end

function test_derivatives(P, u, Omega, uxx, uyy, uxy)
    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, u, Omega)
    @test inn(Array(P.uxx)) ≈ inn(uxx)
    @test inn(Array(P.uyy)) ≈ inn(uyy)
    @test inn(Array(P.uxy)) ≈ inn(uxy)
end

@testset "derivatives" begin
    Omega, P, u, uxx, uyy, uxy = derivative_stdsetup(false)
    test_derivatives(P, u, Omega, uxx, uyy, uxy)
end


#=
# function check_uneven_pseudodiff()
Omega = ComputationDomain(3000e3, 7, correct_distortion = false)
X, Y, K = Omega.X, Omega.Y, Omega.K
# F = sin.(X) + 4 .* cos.(Y) + (X .* Y) .^ 2
F = 2 .* X .+ 4 .* Y
dF = real.(ifft(Omega.pseudodiff .* fft(F)))
# x = Omega.X[:, Omega.My]

OmegaK = ComputationDomain(3000e3, 7, correct_distortion = true)
X, Y, K = OmegaK.X, OmegaK.Y, OmegaK.K
# G = sin.(X .* K) + 4 .* cos.(Y .* K) + (X .* K .* Y .* K) .^ 2
G = 2 .* (X .* K) .+ 4 .* (Y .* K)
dG = real.(ifft(OmegaK.pseudodiff .* fft(G)))

dX = real.(ifft(Omega.pseudodiff .* fft(X)))
dXk = real.(ifft(OmegaK.pseudodiff .* fft(X .* K)))
______________________________________________________________

pd = real.(fft( ifft(Omega.pseudodiff) ./ Omega.K ))
dXK = real.(ifft(pd .* fft(OmegaK.X .* OmegaK.K)))

dX = real.(ifft(Omega.pseudodiff .* fft(Omega.X)))

Xk = OmegaK.K .* OmegaK.X
Yk = OmegaK.K .* OmegaK.Y
xk = vec(Xk)
yk = vec(Yk)
k = vcat(xk', yk') ./ 1e7
ipd = vec(real.(ifft(Omega.pseudodiff)))
nfft(k, ipd)
# dXKK = real.(ifft(Omega.pseudodiff .* fft(Omega.X)))

scaling = fft(OmegaK.X .* OmegaK.K) ./ (fft(Omega.X) .+ 1e3)
=#
