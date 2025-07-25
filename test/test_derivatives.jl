function derivative_stdsetup(use_cuda::Bool)
    W, n = 3000e3, 7
    domain = RegionalDomain(W, n, correct_distortion = false, use_cuda = use_cuda)
    c, p, t_out, _, _, t_Hice, Hice = benchmark1_constants(domain)
    sim = Simulation(domain, c, p, t_out, t_Hice, Hice)

    u = domain.X .^ 2 .* domain.Y .^ 2
    uxx = 2 .* domain.Y .^ 2
    uyy = 2 .* domain.X .^ 2
    uxy = 4 .* domain.X .* domain.Y
    return domain, sim.tools.prealloc, domain.arraykernel(u), uxx, uyy, uxy
end

function test_derivatives(P, u, domain, uxx, uyy, uxy)
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x, P.buffer_xy, u, domain)
    @test inn(Array(P.buffer_xx)) ≈ inn(uxx)
    @test inn(Array(P.buffer_yy)) ≈ inn(uyy)
    @test inn(Array(P.buffer_xy)) ≈ inn(uxy)
end

@testset "derivatives" begin
    domain, P, u, uxx, uyy, uxy = derivative_stdsetup(false)
    test_derivatives(P, u, domain, uxx, uyy, uxy)
end


#=
# function check_uneven_pseudodiff()
domain = RegionalDomain(3000e3, 7, correct_distortion = false)
X, Y, K = domain.X, domain.Y, domain.K
# F = sin.(X) + 4 .* cos.(Y) + (X .* Y) .^ 2
F = 2 .* X .+ 4 .* Y
dF = real.(ifft(domain.pseudodiff .* fft(F)))
# x = domain.X[:, domain.my]

domainK = RegionalDomain(3000e3, 7, correct_distortion = true)
X, Y, K = domainK.X, domainK.Y, domainK.K
# G = sin.(X .* K) + 4 .* cos.(Y .* K) + (X .* K .* Y .* K) .^ 2
G = 2 .* (X .* K) .+ 4 .* (Y .* K)
dG = real.(ifft(domainK.pseudodiff .* fft(G)))

dX = real.(ifft(domain.pseudodiff .* fft(X)))
dXk = real.(ifft(domainK.pseudodiff .* fft(X .* K)))
______________________________________________________________

pd = real.(fft( ifft(domain.pseudodiff) ./ domain.K ))
dXK = real.(ifft(pd .* fft(domainK.X .* domainK.K)))

dX = real.(ifft(domain.pseudodiff .* fft(domain.X)))

Xk = domainK.K .* domainK.X
Yk = domainK.K .* domainK.Y
xk = vec(Xk)
yk = vec(Yk)
k = vcat(xk', yk') ./ 1e7
ipd = vec(real.(ifft(domain.pseudodiff)))
nfft(k, ipd)
# dXKK = real.(ifft(domain.pseudodiff .* fft(domain.X)))

scaling = fft(domainK.X .* domainK.K) ./ (fft(domain.X) .+ 1e3)
=#
