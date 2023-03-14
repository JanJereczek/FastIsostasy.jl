using LinearAlgebra

function mask_disc(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return mask_disc(sqrt.(X.^2 + Y.^2), R)
end

function mask_disc(r::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(r .< R)
end

function uniform_ice_cylinder(
    Omega::ComputationDomain,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R)
    return M .* H
end

function stereo_ice_cylinder(
    Omega::ComputationDomain,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.R, R)
    return M .* H
end

function generate_uniform_disc_load(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R)
    return - M .* (c.rho_ice * c.g * H)
end

function stereo_ice_cap(
    Omega::ComputationDomain,
    alpha_deg::T,
    H::T,
) where {T<:AbstractFloat}
    alpha = deg2rad(alpha_deg)
    M = Omega.Theta .< alpha
    return H .* sqrt.( M .* (cos.(Omega.Theta) .- cos(alpha)) ./ (1 - cos(alpha)) )
end
################################################
# Analytic solution for constant viscosity
################################################
function analytic_solution(
    r::T,
    t::T,
    c::PhysicalConstants,
    p::MultilayerEarth,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=5::Int,
) where {T<:AbstractFloat}
    scaling = c.rho_ice * c.g * H0 * R0
    if t == T(Inf)
        equilibrium_integrand_r(kappa) = equilibrium_integrand(kappa, r, c, p, R0)
        return scaling .* looped_quadrature1D( equilibrium_integrand_r, domains, n_quad_support )
    else
        transient_integrand_r(kappa) = analytic_integrand(kappa, r, t, c, p, R0)
        return scaling .* looped_quadrature1D( transient_integrand_r, domains, n_quad_support )
    end
end

function looped_quadrature1D( 
    f::Function,
    domains::Vector{T},
    n::Int,
) where{T<:Real}
    integral = T(0)
    for i in eachindex(domains)[1:end-1]
        integral += quadrature1D( f, n, domains[i], domains[i+1] )
    end
    return integral
end

function analytic_integrand(
    kappa::T,
    r::T,
    t::T,
    c::PhysicalConstants,
    p::MultilayerEarth,
    R0::T,
) where {T<:AbstractFloat}

    # Here we assume that p-fields are constant over Omega
    beta = mean(p.mean_density) * c.g + mean(p.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    eta = mean(p.effective_viscosity)
    return (exp(-beta*t/(2*eta*kappa))-1) * j0 * j1 / beta
end

function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    p::MultilayerEarth,
    R0::T,
) where {T<:AbstractFloat}
    beta = mean(p.mean_density) * c.g + mean(p.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end

################################################
# Analytic solution for radially dependent viscosity
################################################

function analytic_radial_solution(
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    t::T,
    c::PhysicalConstants,
    p::MultilayerEarth,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=5::Int,
) where {T<:AbstractFloat}
    scaling = c.rho_ice * c.g * H0 * R0
    radial_integrand(kappa) = analytic_radial_integrand(Omega, i, j, kappa, t, c, p, R0)
    return scaling .* looped_quadrature1D( radial_integrand, domains, n_quad_support )
end

function analytic_radial_integrand(
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    kappa::T,
    t::T,
    c::PhysicalConstants,
    p::MultilayerEarth,
    R0::T,
) where {T<:AbstractFloat}

    x, y = Omega.X[i, j], Omega.Y[i, j]
    r = get_r(x, y)

    beta = mean(p.mean_density) * c.g + mean(p.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    return (exp(-beta*t/(2*mean(p.effective_viscosity)*kappa))-1) * j0 * j1 / beta
end

################################################
# Generate binary parameter fields for test 3
################################################

function generate_binary_field(
    Omega::ComputationDomain{T},
    x_lo::T,
    x_hi::T,
) where {T<:AbstractFloat}
    lo_half = fill(x_lo, Omega.N, Int(Omega.N/2))
    hi_half = fill(x_hi, Omega.N, Int(Omega.N/2))
    return cat(lo_half, hi_half, dims=2)
end

function generate_sigmoid_field(
    Omega::ComputationDomain{T},
    x_lo::T,
    x_hi::T,
) where {T<:AbstractFloat}

    delta_x = x_hi - x_lo
    scaled_sigmoid(x) = sigmoid(x, 1e-2)
    return x_lo .+ delta_x .* scaled_sigmoid.(Omega.X)
end
sigmoid(x::Real, c::Real) = 1 / (1 + exp(-c*x))

function generate_window_field(
    Omega::ComputationDomain{T},
    x_background::T,
    x_lo::T,
    x_hi::T,
) where {T<:AbstractFloat}
    N = Omega.N
    N2 = Int(N/2)
    N4 = Int(N/4)
    X = fill(x_background, N, N)
    X[N4+1:end-N4, N4+1:N2] .= x_lo
    X[N4+1:end-N4, N2+1:end-N4] .= x_hi
    return X
end

function generate_gaussian_field(
    Omega::ComputationDomain{T},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::AbstractMatrix{T},
) where {T<:AbstractFloat}
    N = Omega.N
    G = gauss_distr( Omega.X, Omega.Y, xy_peak, sigma )
    G = G ./ maximum(G) .* z_peak
    return fill(z_background, N, N) + G
end