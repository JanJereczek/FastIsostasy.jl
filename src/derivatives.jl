function update_second_derivatives!(uxx, uyy, ux, uxy, u, Omega)
    update_second_derivatives!(uxx, uyy, ux, uxy, u, u, u, Omega)
end

function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, u2::M, u3::M,
    Omega::ComputationDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    dxx!(uxx, u1, Omega)
    dyy!(uyy, u2, Omega)
    dxy!(ux, uxy, u3, Omega)
end

function dxx(u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    du = Matrix{T}(undef, size(u)...)
    dxx!(du, u, Omega)
    return du
end

function dxx!(du::M, u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.nx-1]
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (Omega.Dx[i, j] ^ 2)
        end
        du[1, j] = (u[3, j] - 2*u[2, j] + u[1, j]) / (Omega.Dx[1, j] ^ 2)
        du[Omega.nx, j] = (u[Omega.nx, j] - 2*u[Omega.nx-1, j] + u[Omega.nx-2, j]) /
            (Omega.Dx[Omega.nx, j] ^ 2)
    end
end

function dyy(u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    du = Matrix{T}(undef, size(u)...)
    dyy!(du, u, Omega)
    return du
end

function dyy!(du::M, u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.ny-1]
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (Omega.Dy[i, j] ^ 2)
        end
        du[i, 1] = (u[i, 3] - 2*u[i, 2] + u[i, 1]) / (Omega.Dy[i, 1] ^ 2)
        du[i, Omega.ny] = (u[i, Omega.ny] - 2*u[i, Omega.ny-1] + u[i, Omega.ny-2]) /
            (Omega.Dy[i, Omega.ny] ^ 2)
    end
end

function dxy(u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    ux = Matrix{T}(undef, size(u)...)
    uxy = Matrix{T}(undef, size(u)...)
    dx!(ux, u, Omega)
    dy!(uxy, ux, Omega)
    return uxy
end

function dxy!(ux, uxy, u, Omega)
    dx!(ux, u, Omega)
    dy!(uxy, ux, Omega)
end

function dx!(du::M, u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.nx-1]
            du[i,j] = (u[i+1, j] - u[i-1, j]) / (2 * Omega.Dx[i, j])
        end
        du[1, j] = (u[2, j] - u[1, j]) / Omega.Dx[1, j]
        du[Omega.nx, j] = (u[Omega.nx, j] - u[Omega.nx-1, j]) / Omega.Dx[Omega.nx, j]
    end
end

function dy!(du::M, u::M, Omega::ComputationDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.ny-1]
            du[i, j] = (u[i,j+1] - u[i,j-1]) / (2 * Omega.Dy[i, j])
        end
        du[i, 1] = (u[i, 2] - u[i, 1]) / Omega.Dy[i, 1]
        du[i, Omega.ny] = (u[i, Omega.ny] - u[i, Omega.ny-1]) / Omega.Dy[i, Omega.ny]
    end
end

#####################################################

# Fourier
"""
    get_differential_fourier(W, N2)

Compute the matrices representing the differential operators in the fourier space.
"""
get_differential_fourier(Omega) = get_differential_fourier(Omega.Wx, Omega.Wy, Omega.nx,
    Omega.ny)

function get_differential_fourier(Wx::T, Wy::T, nx::Int, ny::Int) where {T<:Real}
    mu_x = π / Wx
    mu_y = π / Wy
    x_coeffs = mu_x .* fftint(nx)
    y_coeffs = mu_y .* fftint(ny)
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