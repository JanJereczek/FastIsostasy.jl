"""
    analytic_solution(r, t, c, solidearth, H0, R0, domains; n_quad_support)

Return the analytic solution of the bedrock displacement resulting from a
cylindrical ice load with radius `R0` and height `H0` on a flat Earth represented
by an elastic plate overlaying a viscous half space. Parameters are provided in
`c, solidearth`. The points at which the solution is computed are specified by the distance `r`
to the center of the domain. The time at which the solution is computed is specified
by `t`.
"""
function analytic_solution(r::T, t, c::PhysicalConstants, solidearth::SolidEarth,
    H0, R0; n_quad_support=5::Int) where {T<:AbstractFloat}

    support = T.(vcat(1e-14, 10 .^ (-10:0.05:-3), 1.0))     # support vector for quadrature
    scaling = c.rho_ice * c.g * H0 * R0
    if t == T(Inf)
        equilibrium_integrand_r(kappa) = equilibrium_integrand(kappa, r, c, solidearth, R0)
        return scaling .* looped_quadrature1D(equilibrium_integrand_r, support, n_quad_support )
    else
        transient_integrand_r(kappa) = analytic_integrand(kappa, r, t, c, solidearth, R0)
        return scaling .* looped_quadrature1D(transient_integrand_r, support, n_quad_support )
    end
end

function looped_quadrature1D( 
    f::Function,
    domains::Vector{T},
    n::Int,
) where{T<:Real}
    integral = T(0)
    for i in eachindex(domains)[1:end-1]
        integral += quadrature1D(f, n, domains[i], domains[i+1])
    end
    return integral
end

function analytic_integrand(
    kappa::T,
    r::T,
    t::T,
    c,  # PhysicalConstants
    solidearth,  # SolidEarth
    R0::T,
) where {T<:AbstractFloat}

    # Here we assume that solidearth-fields are constant over domain
    beta = solidearth.rho_uppermantle * c.g + mean(solidearth.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    eta = mean(solidearth.effective_viscosity)
    return (exp(-beta*t/(2*eta*kappa))-1) * j0 * j1 / beta
end

function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    solidearth::SolidEarth,
    R0::T,
) where {T<:AbstractFloat}
    beta = solidearth.rho_uppermantle * c.g + mean(solidearth.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end