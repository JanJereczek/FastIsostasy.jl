using FiniteDifferences
# using StaticArraysCore

struct FiniteDiffMethod{T}
    grid::Vector{Int}
    coefs::Vector{T}
end

struct FiniteDiffParams{T}
    order::Int
    r::Int
    d1_central::FiniteDiffMethod{T}
    d1_forward::FiniteDiffMethod{T}
    d1_backward::FiniteDiffMethod{T}
    d2_central::FiniteDiffMethod{T}
    d2_forward::FiniteDiffMethod{T}
    d2_backward::FiniteDiffMethod{T}
end

function FiniteDiffParams(; order = 5, T = Float32)
    r = order ÷ 2
    d1_central = FiniteDifferences.central_fdm(order, 1)
    d1_forward = FiniteDifferences.forward_fdm(order, 1)
    d1_backward = FiniteDifferences.backward_fdm(order, 1)
    d2_central = FiniteDifferences.central_fdm(order, 2)
    d2_forward = FiniteDifferences.forward_fdm(order, 2)
    d2_backward = FiniteDifferences.backward_fdm(order, 2)

    d1_central = FiniteDiffMethod(Vector(d1_central.grid), Vector(T.(d1_central.coefs)))
    d1_forward = FiniteDiffMethod(Vector(d1_forward.grid), Vector(T.(d1_forward.coefs)))
    d1_backward = FiniteDiffMethod(Vector(d1_backward.grid), Vector(T.(d1_backward.coefs)))
    d2_central = FiniteDiffMethod(Vector(d2_central.grid), Vector(T.(d2_central.coefs)))
    d2_forward = FiniteDiffMethod(Vector(d2_forward.grid), Vector(T.(d2_forward.coefs)))
    d2_backward = FiniteDiffMethod(Vector(d2_backward.grid), Vector(T.(d2_backward.coefs)))

    return FiniteDiffParams(
        order,
        r,
        d1_central,
        d1_forward,
        d1_backward,
        d2_central,
        d2_forward,
        d2_backward
    )
end

function update_second_derivatives!(uxx, uyy, ux, uxy, u, domain)
    update_second_derivatives!(uxx, uyy, ux, uxy, u, u, u, domain)
end

function update_second_derivatives!(uxx::M, uyy::M, ux::M, uxy::M, u1::M, u2::M, u3::M,
    domain::RegionalDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    dxx!(uxx, u1, domain)
    dyy!(uyy, u2, domain)
    dxy!(ux, uxy, u3, domain)
end

function dxx(u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    du = Matrix{T}(undef, size(u)...)
    dxx!(du, u, domain)
    return du
end

function dxx!(du::M, u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:domain.nx-1]
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (domain.Dx[i, j] ^ 2)
        end
        du[1, j] = (u[3, j] - 2*u[2, j] + u[1, j]) / (domain.Dx[1, j] ^ 2)
        du[domain.nx, j] = (u[domain.nx, j] - 2*u[domain.nx-1, j] + u[domain.nx-2, j]) /
            (domain.Dx[domain.nx, j] ^ 2)
    end
end

function dxx!(du::M, u::M, domain::RegionalDomain{T, L, M},
    method::FiniteDiffParams{T}) where {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    
    (; r, d2_central, d2_forward, d2_backward) = method
    for j in axes(u, 2)
        for i in axes(u, 1)[r+1:end-r]
            @inbounds for ii in eachindex(d2_central.grid)
                du[i, j] = muladd(d2_central.coefs[ii],
                    u[i + d2_central.grid[ii], j], du[i, j])
            end
        end
        for i in axes(u, 1)[1:r]
            @inbounds for ii in eachindex(d2_forward.grid)
                du[i, j] = muladd(d2_forward.coefs[ii],
                    u[i + d2_forward.grid[ii], j], du[i, j])
            end
        end
        for i in axes(u, 1)[end-r+1:end]
            @inbounds for ii in eachindex(d2_backward.grid)
                du[i, j] = muladd(d2_backward.coefs[ii],
                    u[i + d2_backward.grid[ii], j], du[i, j])
            end
        end
    end
    du ./= (domain.Dx .^ 2)
    return nothing
end


function dyy(u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    du = Matrix{T}(undef, size(u)...)
    dyy!(du, u, domain)
    return du
end

function dyy!(du::M, u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:domain.ny-1]
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (domain.Dy[i, j] ^ 2)
        end
        du[i, 1] = (u[i, 3] - 2*u[i, 2] + u[i, 1]) / (domain.Dy[i, 1] ^ 2)
        du[i, domain.ny] = (u[i, domain.ny] - 2*u[i, domain.ny-1] + u[i, domain.ny-2]) /
            (domain.Dy[i, domain.ny] ^ 2)
    end
end

function dyy!(du::M, u::M, domain::RegionalDomain{T, L, M},
    method::FiniteDiffParams{T}) where {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    
    (; r, d2_central, d2_forward, d2_backward) = method
    @inbounds for i in axes(du, 1)
        @inbounds for j in axes(du, 2)[r+1:end-r]
            @inbounds for jj in eachindex(d2_central.grid)
                du[i, j] = muladd(d2_central.coefs[jj],
                    u[i, j + d2_central.grid[jj]], du[i, j])
            end
        end

        @inbounds for j in axes(du, 2)[1:r]
            @inbounds for jj in eachindex(d2_forward.grid)
                du[i, j] = muladd(d2_forward.coefs[jj],
                    u[i, j + d2_forward.grid[jj]], du[i, j])
            end
        end

        @inbounds for j in axes(du, 2)[end-r+1:end]
            @inbounds for jj in eachindex(d2_backward.grid)
                du[i, j] = muladd(d2_backward.coefs[jj],
                    u[i, j + d2_backward.grid[jj]], du[i, j])
            end
        end
    end

    du ./= (domain.Dy .^ 2)
    return nothing
end

function dxy(u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    ux = Matrix{T}(undef, size(u)...)
    uxy = Matrix{T}(undef, size(u)...)
    dx!(ux, u, domain)
    dy!(uxy, ux, domain)
    return uxy
end

function dxy!(ux, uxy, u, domain)
    dx!(ux, u, domain)
    dy!(uxy, ux, domain)
end

function dx!(du::M, u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:domain.nx-1]
            du[i,j] = (u[i+1, j] - u[i-1, j]) / (2 * domain.Dx[i, j])
        end
        du[1, j] = (u[2, j] - u[1, j]) / domain.Dx[1, j]
        du[domain.nx, j] = (u[domain.nx, j] - u[domain.nx-1, j]) / domain.Dx[domain.nx, j]
    end
end

function dy!(du::M, u::M, domain::RegionalDomain{T, L, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:Matrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:domain.ny-1]
            du[i, j] = (u[i,j+1] - u[i,j-1]) / (2 * domain.Dy[i, j])
        end
        du[i, 1] = (u[i, 2] - u[i, 1]) / domain.Dy[i, 1]
        du[i, domain.ny] = (u[i, domain.ny] - u[i, domain.ny-1]) / domain.Dy[i, domain.ny]
    end
end

#####################################################

# Fourier
"""
$(TYPEDSIGNATURES)

Compute the matrices representing the differential operators in the fourier space.
"""
get_differential_fourier(domain) = get_differential_fourier(domain.Wx, domain.Wy, domain.nx,
    domain.ny)

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