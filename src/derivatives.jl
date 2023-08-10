#######################################################
# 1st order derivative
#######################################################

############################ x ########################
# FDM in y, 1st order derivative, 2nd order convergence
function central_fdx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n1 = size(X, 1)
    return (view(X, 3:n1, :) - view(X, 1:n1-2, :)) ./ (2 .* view(h, 2:n1-1, :))
end

function central_fdx(X::M, h::T) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n1 = size(X, 1)
    return (view(X, 3:n1, :) - view(X, 1:n1-2, :)) ./ (2 * h)
end

# FDM in y, 1st order derivative, 1st order convergence
function forward_fdx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return (view(X, 2, :) - view(X, 1, :)) ./ view(h, 1, :)
end

function backward_fdx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n1 = size(X, 1)
    return (view(X, n1, :) - view(X, n1-1, :)) ./ view(h, n1, :)
end

function mixed_fdx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return cat( forward_fdx(X, h)', central_fdx(X, h), backward_fdx(X, h)', dims=1 )
end

function mixed_fdx!(dudx::M, u::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    m, n = size(dudx)
    @inbounds for j in axes(dudx, 2)
        for i in axes(dudx, 1)[2:m-1]
            dudx[i,j] = (u[i+1,j] - u[i-1,j]) / h[i, j]
        end
        dudx[1, j] = (u[2, j] - u[1, j]) / h[1, j]
        dudx[m, j] = u[m, j] - u[m-1, j] / h[m, j]
    end
end

############################ x ########################
# FDM in x, 1st order derivative, 2nd order convergence
function central_fdy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n2 = size(X, 2)
    return (view(X, :, 3:n2) - view(X, :, 1:n2-2)) ./ (2 .* view(h, :, 2:n2-1))
end

# FDM in x, 1st order derivative, 1st order convergence
function forward_fdy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return (view(X, :, 2) - view(X, :, 1)) ./ view(h, :, 1)
end

function backward_fdy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n2 = size(X, 2)
    return (view(X, :, n2) - view(X, :, n2-1)) ./ view(h, :, n2)
end

function mixed_fdy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return cat( forward_fdy(X, h), central_fdy(X, h), backward_fdy(X, h), dims=2 )
end

#######################################################
# 2nd order derivative
#######################################################

############################ y ########################
# FDM in y, 2nd order
function central_fdxx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n1 = size(X, 1)
    return (view(X, 3:n1, :) - 2 .* view(X, 2:n1-1, :) + view(X, 1:n1-2, :)) ./ 
        view(h, 2:n1-1, :) .^ 2
end

function forward_fdxx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return (view(X, 3, :) - 2 .* view(X, 2, :) + view(X, 1, :)) ./ 
        view(h, 1, :) .^ 2
end

function backward_fdxx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n1 = size(X, 1)
    return (view(X, n1, :) - 2 .* view(X, n1-1, :) + view(X, n1-2, :)) ./ 
        view(h, n1, :) .^ 2
end

function mixed_fdxx(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return cat( forward_fdxx(X,h)', central_fdxx(X,h), backward_fdxx(X,h)', dims=1 )
end

function dxx!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:AbstractMatrix{T}}
    @boundscheck (Omega.Nx, Omega.Ny) == size(u) || throw(BoundsError())
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.Nx-1]
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (Omega.K[i, j] * Omega.dx)^2
        end
        du[1, j] = du[2, j]
        du[Omega.Nx, j] = du[Omega.Nx-1, j]
    end
end

function dyy!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:AbstractMatrix{T}}
    @boundscheck (Omega.Nx, Omega.Ny) == size(u) || throw(BoundsError())
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.Ny-1]
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (Omega.K[i, j] * Omega.dy)^2
        end
        du[i, 1] = du[i, 2]
        du[i, Omega.Ny] = du[i, Omega.Ny-1]
    end
end

function dxy!(du1::M, du2::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:AbstractMatrix{T}}
    # @boundscheck (dxy.Nx, dxy.Ny) == size(u) || throw(BoundsError())
    dx!(du1, u, Omega)
    dy!(du2, du1, Omega)
end

function dx!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:AbstractMatrix{T}}
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:Omega.Nx-1]
            du[i,j] = (u[i+1, j] - u[i-1, j]) / (2 * Omega.K[i, j] * Omega.dx)
        end
        du[1, j] = du[2, j]
        du[Omega.Nx, j] = du[Omega.Nx-1, j]
    end
end

function dy!(du::M, u::M, Omega::ComputationDomain{T, M}) where
    {T<:AbstractFloat, M<:AbstractMatrix{T}}
    @inbounds for i in axes(du, 1)
        for j in axes(du, 2)[2:Omega.Ny-1]
            du[i, j] = (u[i,j+1] - u[i,j-1]) / (2 * Omega.K[i, j] * Omega.dy)
        end
        du[i, 1] = du[i, 2]
        du[i, Omega.Ny] = du[i, Omega.Ny-1]
    end
end

function mixed_fdxy(X::M, hx::M, hy::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return mixed_fdy(mixed_fdx(X, hx), hy)
end

function central_fdxy(X::M, hx::M, hy::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return central_fdy(central_fdx(X, hx), hy)
end

############################ y ########################
# FDM in x, 2nd order derivative, 2nd order convergence
function central_fdyy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n2 = size(X, 2)
    return (view(X, :, 3:n2) - 2 .* view(X, :, 2:n2-1) + view(X, :, 1:n2-2)) ./ 
        view(h, :, 2:n2-1) .^ 2
end

# FDM in x, 2nd order derivative, 1st order convergence
function forward_fdyy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return (view(X, :, 3) - 2 .* view(X, :, 2) + view(X, :, 1)) ./ 
        view(h, :, 1) .^ 2
end

function backward_fdyy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    n2 = size(X, 2)
    return (view(X, :, n2) - 2 .* view(X, :, n2-1) + view(X, :, n2-2)) ./ 
        view(h, :, n2) .^ 2
end

function mixed_fdyy(X::M, h::M) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return cat( forward_fdyy(X,h), central_fdyy(X,h), backward_fdyy(X,h), dims=2 )
end

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