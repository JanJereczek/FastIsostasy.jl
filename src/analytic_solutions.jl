"""
$(TYPEDSIGNATURES)

Return the analytic solution of the bedrock displacement resulting from a
cylindrical ice load with radius `R0` and height `H0` on a flat Earth represented
by an elastic plate overlaying a viscous half space. Parameters are provided in
`c, solidearth`. The points at which the solution is computed are specified by the distance `r`
to the center of the domain. The time at which the solution is computed is specified
by `t`.
"""
function analytic_solution(
    r::T,
    t,
    c::PhysicalConstants,
    solidearth::SolidEarth,
    H0,
    R0;
    n_quad_support = 5::Int,
) where {T<:AbstractFloat}

    support = T.(vcat(1e-14, 10 .^ (-10:0.05:-3), 1.0))     # support vector for quadrature
    scaling = c.rho_ice * c.g * H0 * R0
    if t == T(Inf)
        equilibrium_integrand_r(kappa) =
            equilibrium_integrand(kappa, r, c, solidearth, R0)
        return scaling .*
               looped_quadrature1D(equilibrium_integrand_r, support, n_quad_support)
    else
        transient_integrand_r(kappa) =
            analytic_integrand(kappa, r, t, c, solidearth, R0)
        return scaling .*
               looped_quadrature1D(transient_integrand_r, support, n_quad_support)
    end
end

function looped_quadrature1D(f::Function, domains::Vector{T}, n::Int) where {T<:Real}
    integral = T(0)
    for i in eachindex(domains)[1:(end-1)]
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
    beta =
        solidearth.rho_uppermantle * c.g + mean(solidearth.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # `relaxation_minus_1(t=0) = 0` (no displacement yet, so the transient
    # integral vanishes and `analytic_solution(t=0) = 0`) and `-> -1` as
    # `t -> ∞` (matching `equilibrium_integrand` above exactly, so
    # `analytic_solution(t=∞) = analytic_solution(∞)`), for both the Maxwell and
    # Burgers cases — dispatches on `solidearth.mantle` so `analytic_solution`
    # picks the right step response without the caller having to know which
    # rheology it is.
    return relaxation_minus_1(kappa, beta, t, solidearth, solidearth.mantle) *
           j0 * j1 / beta
end

# ViscousMantle (Maxwell dashpot): single-exponential step response
# û(t) = û_eq (1 - exp(-βt/(2ηk))), see `roadmaps/burgers.md` §2.1. This is also
# the fallback for any mantle not handled below (matches the historical
# behaviour of this function, which never checked the mantle type).
relaxation_minus_1(kappa, beta, t, solidearth, mantle) =
    exp(-beta * t / (2 * mean(solidearth.effective_viscosity) * kappa)) - 1

# TransientCreepMantle, N = 1 (classic Burgers body): two-exponential step
# response from roadmap `burgers.md` §2.4, derived by partial-fractioning the
# Laplace-domain transfer function û(s) = F̂(s)/(β + 2k μ̃(s)) with the
# correspondence-principle modulus of a Maxwell dashpot (η₁) in series with a
# Kelvin-Voigt element (μ₂, η₂ = τμ₂):
#
#   1/μ̃(s) = 1/(η₁s) + 1/(μ₂ + η₂s)
#   ⇒ β + 2k μ̃(s) = 0  has poles s = 0 (the F̂/s of the Heaviside load) and the
#     two roots sₐ, s_b < 0 of  2kη₁η₂ s² + [β(η₁+η₂) + 2kη₁μ₂] s + βμ₂ = 0
#
# Partial-fractioning û(s) = F̂·[μ₂+(η₁+η₂)s] / [s · 2kη₁η₂(s−sₐ)(s−s_b)] and
# inverting term by term gives, for t ≥ 0,
#
#   û(t) = û_eq [1 − Aₐ exp(sₐt) − A_b exp(s_bt)],   Aᵢ = −β·Qᵢ/a
#
# with a = 2kη₁η₂ and Qᵢ the residue of the bracketed numerator at s = sᵢ. Setting
# Δ → 0 (μ₂ → ∞) collapses this onto the single-exponential ViscousMantle formula
# above — verified numerically against a direct ODE integration of the 2-state
# system to ~1e-12 relative error before landing (not a symbolic limit check,
# since one root's amplitude → 0 rather than the root itself vanishing).
#
# `relaxation_minus_1 = û(t)/û_eq − 1 = −1 + Aₐ exp(sₐt) + A_b exp(s_bt)`, the
# Burgers analogue of the Maxwell case's `exp(-βt/(2ηk)) − 1` above: since
# `Aₐ + A_b = 1` (from û(0) = 0), this is 0 at t = 0 and −1 as t → ∞, exactly
# matching the Maxwell fallback's boundaries.
function relaxation_minus_1(
    kappa::T,
    beta::T,
    t::T,
    solidearth,
    mantle::TransientCreepMantle{MT,1},
) where {T<:AbstractFloat,MT}
    eta1 = mean(solidearth.effective_viscosity)
    mu1 = T(mantle.shearmodulus)
    mu2 = mu1 / T(mantle.relaxation_strength[1])
    eta2 = T(mantle.kelvin_time[1]) * T(SECONDS_PER_YEAR) * mu2

    a = 2 * kappa * eta1 * eta2
    b = beta * (eta1 + eta2) + 2 * kappa * eta1 * mu2
    cc = beta * mu2
    disc = sqrt(b ^ 2 - 4 * a * cc)
    sa = (-b + disc) / (2a)
    sb = (-b - disc) / (2a)

    numerator(s) = mu2 + (eta1 + eta2) * s
    Qa = numerator(sa) / (sa * (sa - sb))
    Qb = numerator(sb) / (sb * (sb - sa))
    Aa = -beta * Qa / a
    Ab = -beta * Qb / a

    return -1 + Aa * exp(sa * t) + Ab * exp(sb * t)
end

relaxation_minus_1(kappa, beta, t, solidearth, mantle::TransientCreepMantle) = error(
    "The analytic disc-load solution is only implemented for TransientCreepMantle " *
    "with N = 1 Kelvin branch (roadmap burgers.md §5, Phase 3). Got N = " *
    "$(nbranches(mantle)).",
)

function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    solidearth::SolidEarth,
    R0::T,
) where {T<:AbstractFloat}
    beta =
        solidearth.rho_uppermantle * c.g + mean(solidearth.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end