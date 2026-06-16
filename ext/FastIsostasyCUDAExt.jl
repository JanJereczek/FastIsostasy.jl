module FastIsostasyCUDAExt

using CUDA
using KernelAbstractions: get_backend, synchronize
import FastIsostasy

CUDA.allowscalar(false)

FastIsostasy.cudainfo() = CUDA.versioninfo()

# CUDA FFT plans — overrides the CPU default in tools.jl
FastIsostasy.choose_fft_plans(X::CuArray) = (
    CUDA.CUFFT.plan_fft!(complex.(X)),
    CUDA.CUFFT.plan_ifft!(complex.(X))
)

# GPU derivative dispatches — kernel functions live in derivatives_parallel.jl
function FastIsostasy.update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, ::M, ::M,
    domain::FastIsostasy.RegionalDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T}, M<:CuMatrix{T}}
    backend = get_backend(u1)
    FastIsostasy.dxx_dyy_dx_kernel!(backend)(uxx, uyy, ux, u1, domain.Dx, domain.Dy; ndrange=(domain.nx, domain.ny))
    synchronize(backend)
    FastIsostasy.dy_kernel!(backend)(uxy, ux, domain.Dy; ndrange=(domain.nx, domain.ny))
    synchronize(backend)
end

function FastIsostasy.dxx!(uxx::M, u::M, Dx::M, nx::Int, ny::Int) where {M<:CuMatrix{<:AbstractFloat}}
    backend = get_backend(u)
    FastIsostasy.dxx_kernel!(backend)(uxx, u, Dx; ndrange=(nx, ny))
    synchronize(backend)
    return nothing
end

function FastIsostasy.dyy!(uyy::M, u::M, Dy::M, nx::Int, ny::Int) where {M<:CuMatrix{<:AbstractFloat}}
    backend = get_backend(u)
    FastIsostasy.dyy_kernel!(backend)(uyy, u, Dy; ndrange=(nx, ny))
    synchronize(backend)
    return nothing
end

function FastIsostasy.dxy!(ux::M, uxy::M, u::M, Dx::M, Dy::M, nx::Int, ny::Int) where {M<:CuMatrix{<:AbstractFloat}}
    backend = get_backend(u)
    FastIsostasy.dx_kernel!(backend)(ux, u, Dx; ndrange=(nx, ny))
    synchronize(backend)
    FastIsostasy.dy_kernel!(backend)(uxy, ux, Dy; ndrange=(nx, ny))
    synchronize(backend)
    return nothing
end

function FastIsostasy.dx!(du::M, u::M, Dx::M, nx::Int, ny::Int) where {M<:CuMatrix{<:AbstractFloat}}
    backend = get_backend(u)
    FastIsostasy.dx_kernel!(backend)(du, u, Dx; ndrange=(nx, ny))
    synchronize(backend)
    return nothing
end

function FastIsostasy.dy!(du::M, u::M, Dy::M, nx::Int, ny::Int) where {M<:CuMatrix{<:AbstractFloat}}
    backend = get_backend(u)
    FastIsostasy.dy_kernel!(backend)(du, u, Dy; ndrange=(nx, ny))
    synchronize(backend)
    return nothing
end

# GPU deformation dispatch
function FastIsostasy.thinplate_horizontal_displacement!(u_x::M, u_y::M, u::M,
    litho_thickness::M, domain) where {M<:CuMatrix{<:AbstractFloat}}
    FastIsostasy.dx!(u_x, u, domain.Dx, domain.nx, domain.ny)
    FastIsostasy.dy!(u_y, u, domain.Dx, domain.nx, domain.ny)
    @. u_x *= -litho_thickness / 2
    @. u_y *= -litho_thickness / 2
    return nothing
end

end
