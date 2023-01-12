
@inline function mask_disc(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(X .^ 2 + Y .^ 2 .< R^2)
end

@inline function generate_uniform_disc_load(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R)
    return - M .* (c.ice_density * c.g * H)
end

@inline function generate_cap_load(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    alpha_deg::T,
    H::T,
) where {T<:AbstractFloat}
    R = sqrt.( Omega.X .^ 2 + Omega.Y .^ 2 )
    Theta = R ./ c.r_equator
    alpha = deg2rad(alpha_deg)
    M = Theta .< alpha
    return - (c.ice_density * H * c.g) .* 
           sqrt.( M .* (cos.(Theta) .- cos(alpha)) ./ (1 - cos(alpha)) )
end

################################################
# Analytic solution for constant viscosity
################################################
@inline function analytic_solution(
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=5::Int,
) where {T<:AbstractFloat}
    scaling = c.ice_density * c.g * H0 * R0
    if t == T(Inf)
        equilibrium_integrand_r(kappa) = equilibrium_integrand(kappa, r, c, p, R0)
        return scaling .* looped_quadrature1D( equilibrium_integrand_r, domains, n_quad_support )
    else
        transient_integrand_r(kappa) = analytic_integrand(kappa, r, t, c, p, R0)
        return scaling .* looped_quadrature1D( transient_integrand_r, domains, n_quad_support )
    end
end

@inline function looped_quadrature1D( 
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

    # Here we assume that p-fields are constant over Omega
    beta = p.mantle_density[1,1] * c.g + p.lithosphere_rigidity[1,1] * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    return (exp(-beta*t/(2*p.halfspace_viscosity[1,1]*kappa))-1) * j0 * j1 / beta
end

@inline function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}
    beta = mean(p.mantle_density) * c.g + mean(p.lithosphere_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end

################################################
# Analytic solution for radially dependent viscosity
################################################

@inline function analytic_radial_solution(
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=5::Int,
) where {T<:AbstractFloat}
    scaling = c.ice_density * c.g * H0 * R0
    radial_integrand(kappa) = analytic_radial_integrand(Omega, i, j, kappa, t, c, p, R0)
    return scaling .* looped_quadrature1D( radial_integrand, domains, n_quad_support )
end

@inline function analytic_radial_integrand(
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    kappa::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}

    x, y = Omega.X[i, j], Omega.Y[i, j]
    r = get_r(x, y)
    # mantle_density = p.mantle_density[i, j]
    # lithosphere_rigidity = p.lithosphere_rigidity[i, j]
    # halfspace_viscosity = p.halfspace_viscosity[i, j]

    # beta = mantle_density * c.g + lithosphere_rigidity * kappa ^ 4
    # j0 = besselj0(kappa * r)
    # j1 = besselj1(kappa * R0)
    # return (exp(-beta*t/(2*halfspace_viscosity*kappa))-1) * j0 * j1 / beta

    beta = mean(p.mantle_density) * c.g + mean(p.lithosphere_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    return (exp(-beta*t/(2*mean(p.halfspace_viscosity)*kappa))-1) * j0 * j1 / beta
end
