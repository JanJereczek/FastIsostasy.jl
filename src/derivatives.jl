function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u::M,
    Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    update_second_derivatives!(uxx, uyy, ux, uxy, u, u, u, Omega)
end

function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, u2::M, u3::M,
    Omega::ComputationDomain{T, M}) where {T<:AbstractFloat, M<:Matrix{T}}
    dxx!(uxx, u1, Omega)
    dyy!(uyy, u2, Omega)
    dxy!(ux, uxy, u3, Omega)
end

function dxx(u::M, Omega::ComputationDomain{T, M}) where {T<:AbstractFloat, M<:Matrix{T}}
    du = zeros(T, size(u)...)
    dxx!(du, u, Omega)
    return du
end

function dxx!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    # @boundscheck (Omega.Nx, Omega.Ny) == size(u) || throw(BoundsError())
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.Nx-1]
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (Omega.Dx[i, j] ^ 2)
        end
        du[1, j] = (u[3, j] - 2*u[2, j] + u[1, j]) / (Omega.Dx[1, j] ^ 2)
        du[Omega.Nx, j] = (u[Omega.Nx, j] - 2*u[Omega.Nx-1, j] + u[Omega.Nx-2, j]) /
            (Omega.Dx[Omega.Nx, j] ^ 2)
    end
end

function dyy(u::M, Omega::ComputationDomain{T, M}) where {T<:AbstractFloat, M<:Matrix{T}}
    du = zeros(T, size(u)...)
    dyy!(du, u, Omega)
    return du
end

function dyy!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    # @boundscheck (Omega.Nx, Omega.Ny) == size(u) || throw(BoundsError())
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.Ny-1]
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (Omega.Dy[i, j] ^ 2)
        end
        du[i, 1] = (u[i, 3] - 2*u[i, 2] + u[i, 1]) / (Omega.Dy[i, 1] ^ 2)
        du[i, Omega.Ny] = (u[i, Omega.Ny] - 2*u[i, Omega.Ny-1] + u[i, Omega.Ny-2]) /
            (Omega.Dy[i, Omega.Ny] ^ 2)
    end
end

function dxy(u::M, Omega::ComputationDomain{T, M}) where {T<:AbstractFloat, M<:Matrix{T}}
    ux = zeros(T, size(u)...)
    uxy = zeros(T, size(u)...)
    dx!(ux, u, Omega)
    dy!(uxy, ux, Omega)
    return uxy
end

function dxy!(ux::M, uxy::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    # @boundscheck (dxy.Nx, dxy.Ny) == size(u) || throw(BoundsError())
    dx!(ux, u, Omega)
    dy!(uxy, ux, Omega)
end

function dx!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.Nx-1]
            du[i,j] = (u[i+1, j] - u[i-1, j]) / (2 * Omega.Dx[i, j])
        end
        du[1, j] = (u[2, j] - u[1, j]) / Omega.Dx[1, j]
        du[Omega.Nx, j] = (u[Omega.Nx, j] - u[Omega.Nx-1, j]) / Omega.Dx[Omega.Nx, j]
    end
end

function dy!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.Ny-1]
            du[i, j] = (u[i,j+1] - u[i,j-1]) / (2 * Omega.Dy[i, j])
        end
        du[i, 1] = (u[i, 2] - u[i, 1]) / Omega.Dy[i, 1]
        du[i, Omega.Ny] = (u[i, Omega.Ny] - u[i, Omega.Ny-1]) / Omega.Dy[i, Omega.Ny]
    end
end

#####################################################

# Fourier
"""
    get_differential_fourier(W, N2)

Compute the matrices representing the differential operators in the fourier space.
"""
get_differential_fourier(Omega) = get_differential_fourier(Omega.Wx, Omega.Wy, Omega.Nx,
    Omega.Ny)

function get_differential_fourier(Wx::T, Wy::T, Nx::Int, Ny::Int) where {T<:Real}
    mu_x = π / Wx
    mu_y = π / Wy
    x_coeffs = mu_x .* fftint(Nx)
    y_coeffs = mu_y .* fftint(Ny)
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    harmonic_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(harmonic_coeffs)
    biharmonic_coeffs = harmonic_coeffs .^ 2
    return pseudodiff_coeffs, harmonic_coeffs, biharmonic_coeffs
end

function get_differential_fourier(Wx::T, Wy::T, Nx::Int, Ny::Int, Mx::Int, My::Int, K::M
    ) where {T<:Real, M<:KernelMatrix{T}}
    mu_x = π / Wx
    mu_y = π / Wy
    # kx = vcat(K[Mx:end, My], K[1:Mx-1, My])
    # ky = vcat(K[Mx, My:end], K[Mx, 1:My-1])
    # kx = K[:, My]
    # ky = K[Mx, :]
    # kxx = kx[fftint(Nx) .+ 1]
    # kyy = ky[fftint(Ny) .+ 1]
    x_coeffs = mu_x .* fftint(Nx) # .* (kx ./ maximum(kx))
    y_coeffs = mu_y .* fftint(Ny) # .* (ky ./ maximum(ky))
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    harmonic_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(harmonic_coeffs)
    biharmonic_coeffs = harmonic_coeffs .^ 2
    return pseudodiff_coeffs, harmonic_coeffs, biharmonic_coeffs
end

function fftint(N::Int)
    N2 = N ÷ 2
    if iseven(N)
        return vcat(0:N2, N2-1:-1:1)
    else
        return vcat(0:N2, N2:-1:1)
    end
end