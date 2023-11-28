function check_derivatives()
    Omega, P, u, uxx, uyy, uxy = derivative_stdsetup(false)
    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, u, Omega)
    @test inn(P.uxx) ≈ inn(uxx)
    @test inn(P.uyy) ≈ inn(uyy)
    @test inn(P.uxy) ≈ inn(uxy)
end

function check_gpu_derivatives()
    Omega, P, u, uxx, uyy, uxy = derivative_stdsetup(true)

end

function derivative_stdsetup(use_cuda::Bool)
    W, n, pc = 3000e3, 7, false
    Omega = ComputationDomain(W, n, correct_distortion = pc, use_cuda = use_cuda)
    c, p, _, __, Hcylinder, t_out = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, Hcylinder)

    u = copy(Omega.X .^ 2 .* Omega.Y .^ 2)
    uxx = 2 .* Omega.Y .^ 2
    uyy = 2 .* Omega.X .^ 2
    uxy = 4 .* Omega.X .* Omega.Y
    return Omega, fip.tools.prealloc, u, uxx, uyy, uxy
end

function check_derivatives(P, u, Omega, uxx, uyy, uxy)
    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, u, Omega)
    @test inn(Array(P.uxx)) ≈ inn(uxx)
    @test inn(Array(P.uyy)) ≈ inn(uyy)
    @test inn(Array(P_gpu.uxy)) ≈ inn(uxy)
end

function inn(X)
    return X[2:end-1, 2:end-1]
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
