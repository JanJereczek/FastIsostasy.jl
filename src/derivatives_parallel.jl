# See derivatives.jl for multiple dispatch. GPU kernels use one-sided differences at
# boundaries to match the CPU implementation in derivatives.jl.

@kernel function dxx_kernel!(du, u, Dx)
    ix, iy = @index(Global, NTuple)
    nx = size(u, 1)
    if ix == 1
        du[1, iy] = (u[3, iy] - 2*u[2, iy] + u[1, iy]) / Dx[1, iy]^2
    elseif ix == nx
        du[nx, iy] = (u[nx, iy] - 2*u[nx-1, iy] + u[nx-2, iy]) / Dx[nx, iy]^2
    else
        du[ix, iy] = (u[ix+1, iy] - 2*u[ix, iy] + u[ix-1, iy]) / Dx[ix, iy]^2
    end
end

@kernel function dyy_kernel!(du, u, Dy)
    ix, iy = @index(Global, NTuple)
    ny = size(u, 2)
    if iy == 1
        du[ix, 1] = (u[ix, 3] - 2*u[ix, 2] + u[ix, 1]) / Dy[ix, 1]^2
    elseif iy == ny
        du[ix, ny] = (u[ix, ny] - 2*u[ix, ny-1] + u[ix, ny-2]) / Dy[ix, ny]^2
    else
        du[ix, iy] = (u[ix, iy+1] - 2*u[ix, iy] + u[ix, iy-1]) / Dy[ix, iy]^2
    end
end

@kernel function dx_kernel!(du, u, Dx)
    ix, iy = @index(Global, NTuple)
    nx = size(u, 1)
    if ix == 1
        du[1, iy] = (u[2, iy] - u[1, iy]) / Dx[1, iy]
    elseif ix == nx
        du[nx, iy] = (u[nx, iy] - u[nx-1, iy]) / Dx[nx, iy]
    else
        du[ix, iy] = (u[ix+1, iy] - u[ix-1, iy]) / (2 * Dx[ix, iy])
    end
end

@kernel function dy_kernel!(du, u, Dy)
    ix, iy = @index(Global, NTuple)
    ny = size(u, 2)
    if iy == 1
        du[ix, 1] = (u[ix, 2] - u[ix, 1]) / Dy[ix, 1]
    elseif iy == ny
        du[ix, ny] = (u[ix, ny] - u[ix, ny-1]) / Dy[ix, ny]
    else
        du[ix, iy] = (u[ix, iy+1] - u[ix, iy-1]) / (2 * Dy[ix, iy])
    end
end

# Fused kernel: computes dxx, dyy, and dx in a single pass over u.
# dy (needed for dxy) must remain a separate kernel since dy at (ix,iy)
# depends on dx values at (ix, iy±1) written by other threads.
@kernel function dxx_dyy_dx_kernel!(uxx, uyy, ux, u, Dx, Dy)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(u)
    # dxx
    if ix == 1
        uxx[1, iy] = (u[3, iy] - 2*u[2, iy] + u[1, iy]) / Dx[1, iy]^2
    elseif ix == nx
        uxx[nx, iy] = (u[nx, iy] - 2*u[nx-1, iy] + u[nx-2, iy]) / Dx[nx, iy]^2
    else
        uxx[ix, iy] = (u[ix+1, iy] - 2*u[ix, iy] + u[ix-1, iy]) / Dx[ix, iy]^2
    end
    # dyy
    if iy == 1
        uyy[ix, 1] = (u[ix, 3] - 2*u[ix, 2] + u[ix, 1]) / Dy[ix, 1]^2
    elseif iy == ny
        uyy[ix, ny] = (u[ix, ny] - 2*u[ix, ny-1] + u[ix, ny-2]) / Dy[ix, ny]^2
    else
        uyy[ix, iy] = (u[ix, iy+1] - 2*u[ix, iy] + u[ix, iy-1]) / Dy[ix, iy]^2
    end
    # dx (intermediate for dxy)
    if ix == 1
        ux[1, iy] = (u[2, iy] - u[1, iy]) / Dx[1, iy]
    elseif ix == nx
        ux[nx, iy] = (u[nx, iy] - u[nx-1, iy]) / Dx[nx, iy]
    else
        ux[ix, iy] = (u[ix+1, iy] - u[ix-1, iy]) / (2 * Dx[ix, iy])
    end
end

