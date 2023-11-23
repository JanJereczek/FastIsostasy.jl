# See derivatives.jl for multiple dispatch
function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, u2::M, u3::M,
    Omega::ComputationDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T}, M<:CuMatrix{T}}
    dxx!(uxx, u1, Omega.Dx, Omega.Nx, Omega.Ny)
    dyy!(uyy, u2, Omega.Dy, Omega.Nx, Omega.Ny)
    dxy!(ux, uxy, u3, Omega.Dx, Omega.Dy, Omega.Nx, Omega.Ny)
end

function dxx!(uxx::M, u::M, Dx::M, Nx::Int, Ny::Int) where {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (2:Nx-1, 1:Ny) dxx!(uxx, u, Dx)
    @parallel (1:Ny) flatbcx!(uxx, Nx)
    return nothing
end
@parallel_indices (ix, iy) function dxx!(du, u, Dx)
    du[ix, iy] = (u[ix+1, iy] - 2 * u[ix, iy] + u[ix-1, iy]) / (Dx[ix, iy]^2)
    return nothing
end

function dyy!(uyy::M, u::M, Dy::M, Nx::Int, Ny::Int) where {T<:AbstractFloat, M<:CuMatrix{T}}
    @parallel (1:Nx, 2:Ny-1) dyy!(uyy, u, Dy)
    @parallel (1:Nx) flatbcy!(uyy, Ny)
    return nothing
end
@parallel_indices (ix, iy) function dyy!(du, u, Dy)
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
@parallel_indices (ix, iy) function dx!(du, u, Dx)
    du[ix, iy] = (u[ix+1, iy] - u[ix-1, iy]) / (2 * Dx[ix, iy])
    return nothing
end
@parallel_indices (ix, iy) function dy!(du, u, Dy)
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