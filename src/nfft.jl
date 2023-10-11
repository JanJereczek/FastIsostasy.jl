using NFFT: plan_nfft, adjoint
using FFTW: fft, fftshift

# Working 1D example
n = 30
x = 360/n * (iseven(n) ? (-n÷2:(n÷2-1)) : (-(n-1)÷2:(n-1)÷2))
c = cosd.(x)
yfft = abs.(fftshift(fft(c)))

k = x / 360
ynfft = abs.(nfft(k, complex.(c)))

# Working 2D example
n = 30
x = 360/n * (iseven(n) ? (-n÷2:(n÷2-1)) : (-(n-1)÷2:(n-1)÷2))
y = copy(x)
X, Y = meshgrid(collect(x), collect(y))
C = cosd.(X) .* cosd.(Y)
yfft = abs.(fftshift(fft(C)))

kx = x ./ 360
ky = x ./ 360
KX, KY = meshgrid(kx, ky)
kkx, kky = vec(KX), vec(KY)
kmat = vcat(kkx', kky')
ynfft = abs.(nfft(kmat, complex.(C)))
diff = vec(yfft) - ynfft
extrema(ynfft)
extrema(yfft)

# Example on pseudodiff operator
Omega = ComputationDomain(3000e3, 7)

function get_kappa_dist(Omega, K)
    (;X, Y, Wx, Wy, Nx, Ny) = Omega
    kappa = ifftshift(ifft(fftshift(Omega.pseudodiff)))
    kkx = vec(K .* (X ./ (2 * Wx)) ./ maximum(K) )
    kky = vec(K .* (Y ./ (2 * Wy)) ./ maximum(K) )
    kmat = vcat(kkx', kky')
    ynfft = nfft(kmat, kappa)
    return reshape(ynfft, Nx, Ny)
end
fourierderiv(Z, kappa) = real.(ifft(fft(Z) .* kappa))

kappa0 = Omega.pseudodiff
kappa1 = get_kappa_dist(Omega, ones(Omega.Nx, Omega.Ny))
kappa2 = get_kappa_dist(Omega, Omega.K)

Z = 1e3 .* sin.(Omega.X ./ 0.5e6) .* sin.(Omega.Y ./ 0.5e6)
A = 1 .+ # 1e-6 .* (Omega.Wx .- abs.(Omega.X)) .^ 2 + 1e-6 .* (Omega.Wy .- abs.(Omega.Y)) .^ 2 +
    1e-11 .* (Omega.Wx .- abs.(Omega.X)) .* (Omega.Wy .- abs.(Omega.Y))
Z0 = fourierderiv(Z, kappa0)
Z1 = fourierderiv(Z, kappa1)
Z2 = fourierderiv(Z, kappa2)
A0 = fourierderiv(A, kappa0)
A1 = fourierderiv(A, kappa1)
A2 = fourierderiv(A, kappa2)
