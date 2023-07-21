#######################################################
# 1st order derivative
#######################################################

############################ x ########################
# FDM in x, 1st order derivative, 2nd order convergence
function central_fdx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, 3:n2) - view(M, :, 1:n2-2)) ./ (2 .* view(h, :, 2:n2-1))
end

# FDM in x, 1st order derivative, 1st order convergence
function forward_fdx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return (view(M, :, 2) - view(M, :, 1)) ./ view(h, :, 1)
end

function backward_fdx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, n2) - view(M, :, n2-1)) ./ view(h, :, n2)
end

function mixed_fdx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return cat( forward_fdx(M, h), central_fdx(M, h), backward_fdx(M, h), dims=2 )
end

############################ y ########################
# FDM in y, 1st order derivative, 2nd order convergence
function central_fdy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, 3:n1, :) - view(M, 1:n1-2, :)) ./ (2 .* view(h, 2:n1-1, :))
end

# FDM in y, 1st order derivative, 1st order convergence
function forward_fdy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return (view(M, 2, :) - view(M, 1, :)) ./ view(h, 1, :)
end

function backward_fdy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, n1, :) - view(M, n1-1, :)) ./ view(h, n1, :)
end

function mixed_fdy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return cat( forward_fdy(M,h)', central_fdy(M,h), backward_fdy(M,h)', dims=1 )
end

#######################################################
# 2nd order derivative
#######################################################

############################ x ########################
# FDM in x, 2nd order derivative, 2nd order convergence
function central_fdxx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, 3:n2) - 2 .* view(M, :, 2:n2-1) + view(M, :, 1:n2-2)) ./ 
        view(h, :, 2:n2-1) .^ 2
end

# FDM in x, 2nd order derivative, 1st order convergence
function forward_fdxx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return (view(M, :, 3) - 2 .* view(M, :, 2) + view(M, :, 1)) ./ 
        view(h, :, 1) .^ 2
end

function backward_fdxx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, n2) - 2 .* view(M, :, n2-1) + view(M, :, n2-2)) ./ 
        view(h, :, n2) .^ 2
end

function mixed_fdxx(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return cat( forward_fdxx(M,h), central_fdxx(M,h), backward_fdxx(M,h), dims=2 )
end

############################ y ########################
# FDM in y, 2nd order
function central_fdyy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, 3:n1, :) - 2 .* view(M, 2:n1-1, :) + view(M, 1:n1-2, :)) ./ 
        view(h, 2:n1-1, :) .^ 2
end

function forward_fdyy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return (view(M, 3, :) - 2 .* view(M, 2, :) + view(M, 1, :)) ./ 
        view(h, 1, :) .^ 2
end

function backward_fdyy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, n1, :) - 2 .* view(M, n1-1, :) + view(M, n1-2, :)) ./ 
        view(h, n1, :) .^ 2
end

function mixed_fdyy(M::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    return cat( forward_fdyy(M,h)', central_fdyy(M,h), backward_fdyy(M,h)', dims=1 )
end

function mixed_fdxy(M::AbstractMatrix{T}, hx::AbstractMatrix{T},
    hy::AbstractMatrix{T}) where {T<:AbstractFloat}
    return mixed_fdy(mixed_fdx(M, hx), hy)
end

function central_fdxy(M::AbstractMatrix{T}, hx::AbstractMatrix{T},
    hy::AbstractMatrix{T}) where {T<:AbstractFloat}
    return central_fdy(central_fdx(M, hx), hy)
end

#######################################################
# Periodic
#######################################################
xperiodic_extension(M, n) = cat( view(M, :, n:n), view(M, :, 1:2), dims=2 )
yperiodic_extension(M, n) = cat( view(M, n:n, :), view(M, 1:2, :), dims=1 )

function periodic_fdx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    bcM = xperiodic_extension(M, n2)
    return cat( central_fdx(bcM, h), central_fdx(M, h), central_fdx(bcM, h), dims=2 )
end

function periodic_fdy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    bcM = yperiodic_extension(M, n1)
    return cat( central_fdy(bcM, h), central_fdy(M, h), central_fdy(bcM, h), dims=1 )
end

function periodic_fdxx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    bcM = xperiodic_extension(M, n2)
    return cat( central_fdxx(bcM, h), central_fdxx(M, h), central_fdxx(bcM, h), dims=2 )
end

function periodic_fdyy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    bcM = yperiodic_extension(M, n1)
    return cat( central_fdyy(bcM, h), central_fdyy(M, h), central_fdyy(bcM, h), dims=1 )
end

function periodic_fdxy(M::AbstractMatrix{T}, hx::T, hy::T) where {T<:AbstractFloat}
    return periodic_fdy(periodic_fdx(M, hx), hy)
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