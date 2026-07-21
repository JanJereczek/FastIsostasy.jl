# Finite-difference stencils. Two backends share one API, selected by dispatch:
#
#   - CPU (`Matrix`): plain `@inbounds` loops. Fast (on par with the old `@turbo`)
#     and, unlike `@turbo`, differentiable by Enzyme.
#   - GPU / other (`AbstractMatrix`, e.g. `CuMatrix`): the KernelAbstractions
#     kernels in `derivatives_parallel.jl`, launched via `get_backend(u)`. No
#     CUDA-extension code is needed — the generic method covers every non-`Matrix`
#     array type at runtime.
#
# The `Matrix` methods are strictly more specific, so the CPU path always wins on
# `Matrix` and the KA path handles everything else.

# ---- single-input second derivatives ----------------------------------------

# GPU / generic: `dxx`, `dyy` and `dx` share the same `u`, so one fused kernel
# computes them in a single pass; `dy` (for the mixed derivative) follows because
# it reads the `ux` the fused kernel just wrote.
function update_second_derivatives!(uxx, uyy, ux, uxy, u, domain)
    backend = get_backend(u)
    dxx_dyy_dx_kernel!(backend)(
        uxx,
        uyy,
        ux,
        u,
        domain.Dx,
        domain.Dy;
        ndrange = (domain.nx, domain.ny),
    )
    synchronize(backend)
    dy_kernel!(backend)(uxy, ux, domain.Dy; ndrange = (domain.nx, domain.ny))
    synchronize(backend)
    return nothing
end

# CPU: separate plain-loop passes (fusing buys little for cache-resident loops).
function update_second_derivatives!(uxx::Matrix, uyy, ux, uxy, u, domain)
    update_second_derivatives!(uxx, uyy, ux, uxy, u, u, u, domain)
    return nothing
end

# Distinct-input case (never fused): each derivative dispatches on its array type.
function update_second_derivatives!(uxx, uyy, ux, uxy, u1, u2, u3, domain)
    dxx!(uxx, u1, domain)
    dyy!(uyy, u2, domain)
    dxy!(ux, uxy, u3, domain)
    return nothing
end

# ---- individual stencils -----------------------------------------------------

function dxx(u, domain)
    du = similar(u)
    dxx!(du, u, domain)
    return du
end

function dxx!(du, u, domain)   # GPU / generic
    backend = get_backend(u)
    dxx_kernel!(backend)(du, u, domain.Dx; ndrange = (domain.nx, domain.ny))
    synchronize(backend)
    return nothing
end

function dxx!(du::Matrix, u::Matrix, domain)   # CPU
    nx = domain.nx
    @inbounds for j in axes(du, 2)
        for i = 2:(nx-1)
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (domain.Dx[i, j] ^ 2)
        end
        du[1, j] = (u[3, j] - 2*u[2, j] + u[1, j]) / (domain.Dx[1, j] ^ 2)
        du[nx, j] = (u[nx, j] - 2*u[nx-1, j] + u[nx-2, j]) / (domain.Dx[nx, j] ^ 2)
    end
    return nothing
end

function dyy(u, domain)
    du = similar(u)
    dyy!(du, u, domain)
    return du
end

function dyy!(du, u, domain)   # GPU / generic
    backend = get_backend(u)
    dyy_kernel!(backend)(du, u, domain.Dy; ndrange = (domain.nx, domain.ny))
    synchronize(backend)
    return nothing
end

function dyy!(du::Matrix, u::Matrix, domain)   # CPU
    ny = domain.ny
    @inbounds for j = 2:(ny-1)
        for i in axes(du, 1)
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (domain.Dy[i, j] ^ 2)
        end
    end
    @inbounds for i in axes(du, 1)
        du[i, 1] = (u[i, 3] - 2*u[i, 2] + u[i, 1]) / (domain.Dy[i, 1] ^ 2)
        du[i, ny] = (u[i, ny] - 2*u[i, ny-1] + u[i, ny-2]) / (domain.Dy[i, ny] ^ 2)
    end
    return nothing
end

function dxy(u, domain)
    ux = similar(u)
    uxy = similar(u)
    dxy!(ux, uxy, u, domain)
    return uxy
end

function dxy!(ux, uxy, u, domain)
    dx!(ux, u, domain)
    dy!(uxy, ux, domain)
    return nothing
end

function dx!(du, u, domain)   # GPU / generic
    backend = get_backend(u)
    dx_kernel!(backend)(du, u, domain.Dx; ndrange = (domain.nx, domain.ny))
    synchronize(backend)
    return nothing
end

function dx!(du::Matrix, u::Matrix, domain)   # CPU
    nx = domain.nx
    @inbounds for j in axes(du, 2)
        for i = 2:(nx-1)
            du[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * domain.Dx[i, j])
        end
        du[1, j] = (u[2, j] - u[1, j]) / domain.Dx[1, j]
        du[nx, j] = (u[nx, j] - u[nx-1, j]) / domain.Dx[nx, j]
    end
    return nothing
end

function dy!(du, u, domain)   # GPU / generic
    backend = get_backend(u)
    dy_kernel!(backend)(du, u, domain.Dy; ndrange = (domain.nx, domain.ny))
    synchronize(backend)
    return nothing
end

function dy!(du::Matrix, u::Matrix, domain)   # CPU
    ny = domain.ny
    @inbounds for j = 2:(ny-1)
        for i in axes(du, 1)
            du[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * domain.Dy[i, j])
        end
    end
    @inbounds for i in axes(du, 1)
        du[i, 1] = (u[i, 2] - u[i, 1]) / domain.Dy[i, 1]
        du[i, ny] = (u[i, ny] - u[i, ny-1]) / domain.Dy[i, ny]
    end
    return nothing
end

#####################################################

# Fourier
"""
$(TYPEDSIGNATURES)

Compute the matrices representing the differential operators in the fourier space.
"""
get_differential_fourier(domain) =
    get_differential_fourier(domain.Wx, domain.Wy, domain.nx, domain.ny)

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
        return vcat(0:N2, (N2-1):-1:1)
    else
        return vcat(0:N2, N2:-1:1)
    end
end
