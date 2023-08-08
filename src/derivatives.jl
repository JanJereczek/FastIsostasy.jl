#######################################################
# 1st order derivative
#######################################################

############################ x ########################
# FDM in y, 1st order derivative, 2nd order convergence
function central_fdx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n1 = size(X, 1)
    return (view(X, 3:n1, :) - view(X, 1:n1-2, :)) ./ (2 .* view(h, 2:n1-1, :))
end

# FDM in y, 1st order derivative, 1st order convergence
function forward_fdx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return (view(X, 2, :) - view(X, 1, :)) ./ view(h, 1, :)
end

function backward_fdx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n1 = size(X, 1)
    return (view(X, n1, :) - view(X, n1-1, :)) ./ view(h, n1, :)
end

function mixed_fdx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return cat( forward_fdx(X, h)', central_fdx(X, h), backward_fdx(X, h)', dims=1 )
end

############################ x ########################
# FDM in x, 1st order derivative, 2nd order convergence
function central_fdy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n2 = size(X, 2)
    return (view(X, :, 3:n2) - view(X, :, 1:n2-2)) ./ (2 .* view(h, :, 2:n2-1))
end

# FDM in x, 1st order derivative, 1st order convergence
function forward_fdy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return (view(X, :, 2) - view(X, :, 1)) ./ view(h, :, 1)
end

function backward_fdy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n2 = size(X, 2)
    return (view(X, :, n2) - view(X, :, n2-1)) ./ view(h, :, n2)
end

function mixed_fdy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return cat( forward_fdy(X, h), central_fdy(X, h), backward_fdy(X, h), dims=2 )
end

#######################################################
# 2nd order derivative
#######################################################

############################ y ########################
# FDM in y, 2nd order
function central_fdxx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n1 = size(X, 1)
    return (view(X, 3:n1, :) - 2 .* view(X, 2:n1-1, :) + view(X, 1:n1-2, :)) ./ 
        view(h, 2:n1-1, :) .^ 2
end

function forward_fdxx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return (view(X, 3, :) - 2 .* view(X, 2, :) + view(X, 1, :)) ./ 
        view(h, 1, :) .^ 2
end

function backward_fdxx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n1 = size(X, 1)
    return (view(X, n1, :) - 2 .* view(X, n1-1, :) + view(X, n1-2, :)) ./ 
        view(h, n1, :) .^ 2
end

function mixed_fdxx(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return cat( forward_fdxx(X,h)', central_fdxx(X,h), backward_fdxx(X,h)', dims=1 )
end

function mixed_fdxy(X::M, hx::M, hy::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return mixed_fdy(mixed_fdx(X, hx), hy)
end

function central_fdxy(X::M, hx::M, hy::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return central_fdy(central_fdx(X, hx), hy)
end

############################ y ########################
# FDM in x, 2nd order derivative, 2nd order convergence
function central_fdyy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n2 = size(X, 2)
    return (view(X, :, 3:n2) - 2 .* view(X, :, 2:n2-1) + view(X, :, 1:n2-2)) ./ 
        view(h, :, 2:n2-1) .^ 2
end

# FDM in x, 2nd order derivative, 1st order convergence
function forward_fdyy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return (view(X, :, 3) - 2 .* view(X, :, 2) + view(X, :, 1)) ./ 
        view(h, :, 1) .^ 2
end

function backward_fdyy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    n2 = size(X, 2)
    return (view(X, :, n2) - 2 .* view(X, :, n2-1) + view(X, :, n2-2)) ./ 
        view(h, :, n2) .^ 2
end

function mixed_fdyy(X::M, h::M) where {M<:AbstractMatrix{<:AbstractFloat}}
    return cat( forward_fdyy(X,h), central_fdyy(X,h), backward_fdyy(X,h), dims=2 )
end

#######################################################
# Periodic
#######################################################
xperiodic_extension(X, n) = cat( view(X, n:n, :), view(X, 1:2, :), dims=1 )
yperiodic_extension(X, n) = cat( view(X, :, n:n), view(X, :, 1:2), dims=2 )

function periodic_fdx(X::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(X, 2)
    bcM = xperiodic_extension(X, n2)
    return cat( central_fdx(bcM, h), central_fdx(X, h), central_fdx(bcM, h), dims=1 )
end

function periodic_fdy(X::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(X, 1)
    bcM = yperiodic_extension(X, n1)
    return cat( central_fdy(bcM, h), central_fdy(X, h), central_fdy(bcM, h), dims=2 )
end

function periodic_fdyy(X::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(X, 2)
    bcM = xperiodic_extension(X, n2)
    return cat( central_fdyy(bcM, h), central_fdyy(X, h), central_fdyy(bcM, h), dims=2 )
end

function periodic_fdxx(X::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(X, 1)
    bcM = yperiodic_extension(X, n1)
    return cat( central_fdxx(bcM, h), central_fdxx(X, h), central_fdxx(bcM, h), dims=1 )
end

function periodic_fdxy(X::AbstractMatrix{T}, hx::T, hy::T) where {T<:AbstractFloat}
    return periodic_fdy(periodic_fdx(X, hx), hy)
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