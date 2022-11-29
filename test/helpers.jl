
function mask_disc(X::Matrix{T}, Y::Matrix{T}, R::T) where {T<:AbstractFloat}
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

function analytic_solution(
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    H0::T,
    R0::T,
) where {T<:AbstractFloat}
    scaling = c.rho_ice * c.g * H0 * R0
    integrand_rt(kappa) = analytic_integrand(kappa, r, t, c, p, R0)
    # lines(0:0.01:1, integrand_rt.(0f0:1f-2:1f0))
    return scaling .* quadrature1D( integrand_rt, 10_000_000, T(0), T(1) )
end

function analytic_integrand(
    kappa::T,
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}
    beta = p.rho_mantle * c.g + p.lithosphere_rigidity * kappa ^ 4
    j1 = besselj1(kappa * R0)
    j0 = besselj0(kappa * r)
    return (exp(-beta*t/(2*p.mantle_viscosity*kappa))-1) * j0 * j1 / beta
end