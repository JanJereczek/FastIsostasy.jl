
function mask_disc(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(X .^ 2 + Y .^ 2 .< R^2)
end

function generate_uniform_disc_load(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    D = mask_disc(Omega.X, Omega.Y, R)
    return -D .* (c.rho_ice * c.g * H)
end

@inline function analytic_solution(
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=1_000::Int,
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

@inline function analytic_integrand(
    kappa::T,
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}
    beta = p.rho_mantle * c.g + p.lithosphere_rigidity * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    return (exp(-beta*t/(2*p.mantle_viscosity*kappa))-1) * j0 * j1 / beta
end

@inline function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}
    beta = p.rho_mantle * c.g + p.lithosphere_rigidity * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end