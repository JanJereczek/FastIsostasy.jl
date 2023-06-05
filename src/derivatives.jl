
# FDM in x, 1st order
function central_fdx(M::XMatrix, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, 3:n2) - view(M, :, 1:n2-2)) ./ (2*h)
end

function forward_fdx(M::XMatrix, h::T) where {T<:AbstractFloat}
    return (view(M, :, 2) - view(M, :, 1)) ./ h
end

function backward_fdx(M::XMatrix, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, n2) - view(M, :, n2-1)) ./ h
end

# FDM in y, 1st order
function mixed_fdx(M::XMatrix, h::T) where {T<:AbstractFloat}
    return cat( forward_fdx(M,h), central_fdx(M,h), backward_fdx(M,h), dims=2 )
end

function central_fdy(M::XMatrix, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, 3:n1, :) - view(M, 1:n1-2, :)) ./ (2*h)
end

function forward_fdy(M::XMatrix, h::T) where {T<:AbstractFloat}
    return (view(M, 2, :) - view(M, 1, :)) ./ h
end

function backward_fdy(M::XMatrix, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, n1, :) - view(M, n1-1, :)) ./ h
end

function mixed_fdy(M::XMatrix, h::T) where {T<:AbstractFloat}
    return cat( forward_fdy(M,h)', central_fdy(M,h), backward_fdy(M,h)', dims=1 )
end

# FDM in x, 2nd order
function central_fdxx(M::XMatrix, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, 3:n2) - 2 .* view(M, :, 2:n2-1) + view(M, :, 1:n2-2)) ./ h^2
end

function forward_fdxx(M::XMatrix, h::T) where {T<:AbstractFloat}
    return (view(M, :, 3) - 2 .* view(M, :, 2) + view(M, :, 1)) ./ h^2
end

function backward_fdxx(M::XMatrix, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, n2) - 2 .* view(M, :, n2-1) + view(M, :, n2-2)) ./ h^2
end

function mixed_fdxx(M::XMatrix, h::T) where {T<:AbstractFloat}
    return cat( forward_fdxx(M,h), central_fdxx(M,h), backward_fdxx(M,h), dims=2 )
end

# FDM in y, 2nd order
function central_fdyy(M::XMatrix, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, 3:n1, :) - 2 .* view(M, 2:n1-1, :) + view(M, 1:n1-2, :)) ./ h^2
end

function forward_fdyy(M::XMatrix, h::T) where {T<:AbstractFloat}
    return (view(M, 3, :) - 2 .* view(M, 2, :) + view(M, 1, :)) ./ h^2
end

function backward_fdyy(M::XMatrix, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, n1, :) - 2 .* view(M, n1-1, :) + view(M, n1-2, :)) ./ h^2
end

function mixed_fdyy(M::XMatrix, h::T) where {T<:AbstractFloat}
    return cat( forward_fdyy(M,h)', central_fdyy(M,h), backward_fdyy(M,h)', dims=1 )
end

function mixed_fdxy(M::XMatrix, hx::T, hy::T) where {T<:AbstractFloat}
    return mixed_fdy(mixed_fdx(M, hx), hy)
end

function gauss_distr(x::T, mu::Vector{T}, sigma::Matrix{T}) where {T<:AbstractFloat}
    k = length(mu)
    return (2 * π)^(k/2) * det(sigma) * exp( -0.5 * (x .- mu)' * inv(sigma) * (x .- mu) )
end


# Fourier
"""
    get_differential_fourier(W, N2)

Compute the matrices representing the differential operators in the fourier space.
"""
function get_differential_fourier(
    W::T,
    N2::Int,
) where {T<:Real}
    mu = T(π / W)
    raw_coeffs = mu .* T.( vcat(0:N2, N2-1:-1:1) )
    x_coeffs, y_coeffs = raw_coeffs, raw_coeffs
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    harmonic_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(harmonic_coeffs)
    biharmonic_coeffs = harmonic_coeffs .^ 2
    return pseudodiff_coeffs, harmonic_coeffs, biharmonic_coeffs
end

function precomp_fourier_dxdy(
    M::XMatrix,
    L1::T,
    L2::T,
) where {T<:AbstractFloat}
    n1, n2 = size(M)
    k1 = 2 * π / L1 .* vcat(0:n1/2-1, 0, -n1/2+1:-1)
    k2 = 2 * π / L2 .* vcat(0:n2/2-1, 0, -n2/2+1:-1)
    p1, p2 = plan_fft(k1), plan_fft(k2)
    ip1, ip2 = plan_ifft(k1), plan_ifft(k2)
    return k1, k2, p1, p2, ip1, ip2
end

function fourier_dnx(
    M::XMatrix,
    k1::Vector{T},
    p1::AbstractFFTs.Plan,
    ip1::AbstractFFTs.ScaledPlan,
) where {T<:AbstractFloat}
    return vcat( [real.(ip1 * ( ( im .* k1 ) .^ n .* (p1 * M[i, :]) ))' for i in axes(M,1)]... )
end