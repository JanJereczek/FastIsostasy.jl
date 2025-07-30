# See derivatives.jl for multiple dispatch
function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, u2::M, u3::M,
    domain::RegionalDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T}, M<:CuMatrix{T}}
    dxx!(uxx, u1, domain.Dx, domain.nx, domain.ny)
    dyy!(uyy, u2, domain.Dy, domain.nx, domain.ny)
    dxy!(ux, uxy, u3, domain.Dx, domain.Dy, domain.nx, domain.ny)
end

function dxx!(uxx::M, u::M, Dx::M, nx::Int, ny::Int) where {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (2:nx-1, 1:ny) dxx!(uxx, u, Dx)
    @parallel (1:ny) flatbcx!(uxx, nx)
    return nothing
end
@parallel_indices (ix, iy) function dxx!(du, u, Dx)
    du[ix, iy] = (u[ix+1, iy] - 2 * u[ix, iy] + u[ix-1, iy]) / (Dx[ix, iy]^2)
    return nothing
end

function dyy!(uyy::M, u::M, Dy::M, nx::Int, ny::Int) where {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (1:nx, 2:ny-1) dyy!(uyy, u, Dy)
    @parallel (1:nx) flatbcy!(uyy, ny)
    return nothing
end
@parallel_indices (ix, iy) function dyy!(du, u, Dy)
    du[ix, iy] = (u[ix, iy+1] - 2 * u[ix, iy] + u[ix, iy-1]) / (Dy[ix, iy]^2)
    return nothing
end

function dxy!(ux::M, uxy::M, u::M, Dx::M, Dy::M, nx::Int, ny::Int) where
    {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (2:nx-1, 1:ny) dx!(ux, u, Dx)
    @parallel (1:ny) flatbcx!(ux, nx)
    @parallel (1:nx, 2:ny-1) dy!(uxy, ux, Dy)
    @parallel (1:nx) flatbcy!(uxy, ny)
    return nothing
end

function dx!(du, u, Dx, nx, ny)
    @parallel (2:nx-1, 1:ny) dx!(du, u, Dx)
    @parallel (1:ny) flatbcx!(du, nx)
    return nothing
end

function dy!(du, u, Dy, nx, ny)
    @parallel (1:nx, 2:ny-1) dy!(du, u, Dy)
    @parallel (1:nx) flatbcy!(du, ny)
    return nothing
end

@parallel_indices (ix, iy) function dx!(du, u, Dx)
    du[ix, iy] = (u[ix+1, iy] - u[ix-1, iy]) / (2 * Dx[ix, iy])
    return nothing
end

@parallel_indices (ix, iy) function dy!(du, u, Dy)
    du[ix, iy] = (u[ix, iy+1] - u[ix, iy-1]) / (2 * Dy[ix, iy])
    return nothing
end

@parallel_indices (iy) function flatbcx!(u, nx)
    u[1, iy] = u[2, iy]
    u[nx, iy] = u[nx-1, iy]
    return nothing
end

@parallel_indices (ix) function flatbcy!(u, ny)
    u[ix, 1] = u[ix, 2]
    u[ix, ny] = u[ix, ny-1]
    return nothing
end