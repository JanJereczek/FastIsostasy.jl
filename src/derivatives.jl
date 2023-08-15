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

function dxx!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    # @boundscheck (Omega.Nx, Omega.Ny) == size(u) || throw(BoundsError())
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.Nx-1]
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (Omega.K[i, j] * Omega.dx)^2
        end
        du[1, j] = du[2, j]
        du[Omega.Nx, j] = du[Omega.Nx-1, j]
    end
end

function dyy!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    # @boundscheck (Omega.Nx, Omega.Ny) == size(u) || throw(BoundsError())
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.Ny-1]
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (Omega.K[i, j] * Omega.dy)^2
        end
        du[i, 1] = du[i, 2]
        du[i, Omega.Ny] = du[i, Omega.Ny-1]
    end
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
            du[i,j] = (u[i+1, j] - u[i-1, j]) / (2 * Omega.K[i, j] * Omega.dx)
        end
        du[1, j] = du[2, j]
        du[Omega.Nx, j] = du[Omega.Nx-1, j]
    end
end

function dy!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.Ny-1]
            du[i, j] = (u[i,j+1] - u[i,j-1]) / (2 * Omega.K[i, j] * Omega.dy)
        end
        du[i, 1] = du[i, 2]
        du[i, Omega.Ny] = du[i, Omega.Ny-1]
    end
end

################

function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, u2::M, u3::M,
    Omega::ComputationDomain{T, M}) where {T<:AbstractFloat, M<:CuMatrix{T}}
    dxx!(uxx, u1, Omega.Dx, Omega.Nx, Omega.Ny)
    dyy!(uyy, u2, Omega.Dy, Omega.Nx, Omega.Ny)
    dxy!(ux, uxy, u3, Omega.Dx, Omega.Dy, Omega.Nx, Omega.Ny)
end

function dxx!(uxx::M, u::M, Dx::M, Nx::Int, Ny::Int) where {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (2:Nx-1, 1:Ny) dxx!(uxx, u, Dx)
    @parallel (1:Ny) flatbcx!(uxx, Nx)
    return nothing
end
@parallel_indices (ix, iy) function dxx!(du::M, u::M, Dx::M) where M
    du[ix, iy] = (u[ix+1, iy] - 2 * u[ix, iy] + u[ix-1, iy]) / (Dx[ix, iy]^2)
    return nothing
end

function dyy!(uyy::M, u::M, Dy::M, Nx::Int, Ny::Int) where {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (1:Nx, 2:Ny-1) dyy!(uyy, u, Dy)
    @parallel (1:Nx) flatbcy!(uyy, Ny)
    return nothing
end
@parallel_indices (ix, iy) function dyy!(du::M, u::M, Dy::M) where M
    du[ix, iy] = (u[ix, iy+1] - 2 * u[ix, iy] + u[ix, iy-1]) / (Dy[ix, iy]^2)
    return nothing
end

function dxy!(ux::M, uxy::M, u::M, Dx::M, Dy::M, Nx::Int, Ny::Int) where
    {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (2:Nx-1, 1:Ny) dx!(ux, u, Dx)
    @parallel (1:Ny) flatbcx!(ux, Nx)
    @parallel (1:Nx, 2:Ny-1) dy!(uxy, ux, Dy)
    @parallel (1:Nx) flatbcy!(uxy, Ny)
    return nothing
end
@parallel_indices (ix, iy) function dx!(du::M, u::M, Dx::M) where M
    du[ix, iy] = (u[ix+1, iy] - u[ix-1, iy]) / (2 * Dx[ix, iy])
    return nothing
end
@parallel_indices (ix, iy) function dy!(du::M, u::M, Dy::M) where M
    du[ix, iy] = (u[ix, iy+1] - u[ix, iy-1]) / (2 * Dy[ix, iy])
    return nothing
end


@parallel_indices (iy) function flatbcx!(u, Nx)
    u[1, iy] = u[2, iy]
    u[Nx, iy] = u[Nx-1, iy]
    return nothing
end

@parallel_indices (ix) function flatbcy!(u, Ny)
    u[ix, 1] = u[ix, 2]
    u[ix, Ny] = u[ix, Ny-1]
    return nothing
end
#####################################################

# Fourier
"""
    get_differential_fourier(W, N2)

Compute the matrices representing the differential operators in the fourier space.
"""
get_differential_fourier(Omega) = get_differential_fourier(Omega.Wx, Omega.Wy, Omega.Nx, Omega.Ny)

function get_differential_fourier(Wx::T, Wy::T, Nx::Int, Ny::Int) where {T<:Real}
    mu_x = π / Wx
    mu_y = π / Wy
    x_coeffs = mu_x .* fftint(Nx)
    y_coeffs = mu_y .* fftint(Ny)
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    harmonic_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(harmonic_coeffs)
    # biharmonic_coeffs = harmonic_coeffs .^ 2
    return pseudodiff_coeffs, harmonic_coeffs
end

function fftint(N::Int)
    N2 = N ÷ 2
    if iseven(N)
        return vcat(0:N2, N2-1:-1:1)
    else
        return vcat(0:N2, N2:-1:1)
    end
end