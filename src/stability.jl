#####################################################
# Explicit-RK stability diagnostics (roadmap: stabilise_dt.md)
#
# Two independent tools used by Phase 0 of the dt-stabilisation roadmap (and
# reused by the RKC stepper of Phase 2 for its own spectral-radius estimate):
#
#   - `real_axis_stability_limit`: the largest β such that a tableau's
#     stability function stays inside the unit disk for z in [-β, 0] — "how
#     many multiples of 1/λ_max can dt be" for a given explicit RK method.
#   - `spectral_radius_estimate`: a nonlinear power iteration (finite-
#     difference Jacobian-vector products, no Jacobian ever formed) estimating
#     the dominant |eigenvalue| of a generic in-place RHS `f!(du, u, p, t)`.
#     Same technique used by RKC/ROCK-type integrators (Sommeijer, Shampine &
#     Verwer 1997) to size their stage count without assembling a Jacobian.
#####################################################

"""
$(TYPEDSIGNATURES)

Stability function `R(z)` of an explicit Runge-Kutta tableau at the scalar
linear test equation `u' = λu` (`z = λ*dt`), computed by the same forward
stage recursion `perform_step!` uses, specialised to a scalar/linear RHS.
"""
function stability_function(z, tab::RKTableau{T}) where {T}
    s = nstages(tab)
    Y = zeros(promote_type(T, typeof(z)), s)
    Y[1] = one(z)
    @inbounds for i = 2:s
        acc = one(z)
        for j = 1:(i-1)
            a = tab.A[i, j]
            iszero(a) && continue
            acc += z * a * Y[j]
        end
        Y[i] = acc
    end
    R = one(z)
    @inbounds for i = 1:s
        b = tab.b[i]
        iszero(b) && continue
        R += z * b * Y[i]
    end
    return R
end

"""
$(TYPEDSIGNATURES)

Largest `β > 0` such that `|R(-β')| <= 1` for every `β' ∈ [0, β]`, i.e. the
real-axis stability boundary of `tab`. Assumes stability holds continuously
from the origin up to the first crossing — true for every tableau currently
defined in `integrators.jl` (`EulerIntegrator`, `BS3Integrator`, `Tsit5Integrator`).
"""
function real_axis_stability_limit(tab::RKTableau{T}; tol = sqrt(eps(T))) where {T}
    stable(β) = abs(stability_function(-β, tab)) <= 1 + sqrt(eps(T))
    lo, hi = zero(T), T(4 * nstages(tab)^2)
    while stable(hi)
        hi *= 2
    end
    while hi - lo > tol * max(one(T), hi)
        mid = (lo + hi) / 2
        stable(mid) ? (lo = mid) : (hi = mid)
    end
    return lo
end

# Computed once at package load. A method's tableau — hence its stability
# boundary — does not depend on the integrator's settings, so the probe
# instances below are built with arbitrary ones.
const _STABILITY_LIMIT_EULER =
    real_axis_stability_limit(tableau(EulerIntegrator(dt = 1.0), Float64))
const _STABILITY_LIMIT_BS3 =
    real_axis_stability_limit(tableau(BS3Integrator(), Float64))
const _STABILITY_LIMIT_TSIT5 =
    real_axis_stability_limit(tableau(Tsit5Integrator(), Float64))

"""
$(TYPEDSIGNATURES)

Real-axis stability limit of an integrator, computed once at package load.

A property of the method's tableau, not of how it is configured, so it takes the
*type* — `stability_limit(Tsit5Integrator)` — and an instance is accepted as a
convenience. Defined for the tableau methods only; `RKCIntegrator` has no fixed
boundary, its stability interval grows with the per-step stage count (see
`rkc_stability_boundary`).
"""
stability_limit(::Type{<:EulerIntegrator}) = _STABILITY_LIMIT_EULER
stability_limit(::Type{<:BS3Integrator}) = _STABILITY_LIMIT_BS3
stability_limit(::Type{<:Tsit5Integrator}) = _STABILITY_LIMIT_TSIT5
stability_limit(alg::AbstractIntegrator) = stability_limit(typeof(alg))

"""
$(TYPEDSIGNATURES)

Estimate the spectral radius (dominant `|eigenvalue|`) of the Jacobian of the
in-place RHS `f!(du, u, p, t)` at `(u0, p, t)`, via nonlinear power iteration
(Sommeijer, Shampine & Verwer 1997): repeatedly evaluate `f!` at a small
perturbation of `u0` along the current direction `v`, use the finite difference
`(f(u0 + h*v) - f(u0)) / h` as the next direction, and take its norm as the
spectral-radius estimate. No Jacobian is ever formed; cost is `O(maxiter)`
extra RHS evaluations. Works for any `AbstractArray` `u0` (CPU or GPU).

`u0` and `p` are never mutated: `f!` is only ever called on internal copies.
Falls back to an alternating-sign pattern (not a constant field) when `F0`
vanishes: for RHSs that end in a mean-subtracting BC (`apply_bc!`/`OffsetBC`,
`src/boundary_conditions.jl`), a spatially uniform probe direction lies exactly
in that operator's null space and the iteration would converge to a spurious
zero instead of the true (generically high-wavenumber) dominant eigenvalue.
"""
function spectral_radius_estimate(
    f!,
    u0,
    p,
    t;
    v0 = nothing,
    maxiter::Int = 100,
    tol = 1e-2,
)

    T = eltype(u0)
    u = copy(u0)
    F0 = similar(u)
    f!(F0, u, p, t)

    v = v0 === nothing ? copy(F0) : copy(v0)
    unorm = norm(u)
    vnorm = norm(v)
    if vnorm < sqrt(eps(T)) * max(one(T), unorm)
        alternating = T.(1 .- 2 .* isodd.(LinearIndices(size(u))))
        v = similar(u);
        copyto!(v, alternating)
        vnorm = norm(v)
    end

    z = similar(u)
    Fz = similar(F0)
    h = sqrt(eps(T)) * max(one(T), unorm)
    sigma = zero(T)

    for _ = 1:maxiter
        @. z = u + (h / vnorm) * v
        f!(Fz, z, p, t)
        @. Fz -= F0
        sigma_new = norm(Fz) / h
        converged = sigma_new > 0 && abs(sigma_new - sigma) <= T(tol) * sigma_new
        sigma = sigma_new
        converged && break
        v = copy(Fz)
        vnorm = norm(v)
        vnorm < sqrt(eps(T)) * max(one(T), sigma) && break
    end
    return sigma
end

"""
$(TYPEDSIGNATURES)

Wrap any `f!(du, u, p, t)` that mutates a [`Simulation`](@ref) `sim` beyond
`du` into a closure safe for repeated, throwaway evaluation — the pattern
`spectral_radius_estimate`'s power iteration needs (and `RKCIntegrator`'s own
spectral-radius estimator, `init_integrator`/`maybe_reestimate!` in
`src/integrators.jl`, reuses for exactly the same reason).

A `Simulation`-backed RHS like `update_diagnostics!` mutates far more than just
`du`: `update_bedrock!` writes the trial `u` straight into `sim.now.u`, and the
sparse-diagnostics block (elastic response, barystatic sea level,
ocean/grounded masks — normally only recomputed once per
`dt_sparse_diagnostics` window, gated by `sim.now.count_sparse_updates`)
mutates several more `sim.now`/`sim.sealevel.bsl` fields. Left unguarded, a
multi-evaluation probe like the power iteration would (a) permanently corrupt
the live simulation's state with whatever trial point was last evaluated —
including the very `u0` array the probe was seeded from, since `update_bedrock!`
writes into `sim.now.u` in place and callers typically pass `sim.now.u` as
`u0` — and (b) only let the gated block run on the *first* evaluation at a
given `t`, contaminating later finite differences with a spurious jump
unrelated to the true Jacobian-vector product. This closure snapshots `sim`'s
state once at construction (via the existing
[`StateSnapshot`](@ref)/[`snapshot!`](@ref)/[`restore!`](@ref) checkpointing
machinery, `src/snapshot.jl`) and restores it after every call, so `sim` is
left exactly as found regardless of how many trial states are evaluated, and
every call sees the gated block exactly as it would at the snapshot time.
"""
function snapshotting_probe(f!, sim::Simulation)
    buf = StateSnapshot(sim)
    # `sim` is injected in the third slot regardless of whatever `p` the caller
    # threads through `spectral_radius_estimate`: `Simulation`-backed RHS's like
    # `update_diagnostics!(dudt, u, sim, t)` take `sim` there by convention, not
    # a generic parameter object, and callers (e.g. `simulation_rhs_probe`) may
    # legitimately pass `p = nothing` since it is otherwise unused.
    return (du, u, _, t) -> begin
        f!(du, u, sim, t)
        restore!(sim, buf)
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Wrap `sim`'s own [`update_diagnostics!`](@ref) with [`snapshotting_probe`](@ref)
— the specific instance `stiffness_report` and the Phase-0 diagnostics use.
"""
simulation_rhs_probe(sim::Simulation) = snapshotting_probe(update_diagnostics!, sim)

"""
$(TYPEDSIGNATURES)

Worst-case analytic cross-check for the laterally-variable Maxwell RHS
(roadmap `stabilise_dt.md` §2.1): evaluates the U-shaped per-mode decay rate

    λ(k) = (ρ g + D k⁴) / (2 η k s_ν)

at the two candidate extrema of the spectrum — the regularised DC wavenumber
`domain.pseudodiff[1,1]` and the largest resolved wavenumber `k_max` — using
pointwise worst-case coefficients (`D = max(litho_rigidity)`,
`η = min(effective_viscosity)`, `s_ν = min(pseudodiff_scaling)`). This ignores
the smoothing convolution applied to the real RHS, so it over-estimates the
true spectral radius; it is a cross-check for
[`spectral_radius_estimate`](@ref), not a replacement.
"""
function analytic_lambda_bound(sim::Simulation)
    se, domain, g = sim.solidearth, sim.domain, sim.c.g
    D = maximum(se.litho_rigidity)
    eta = minimum(se.effective_viscosity)
    s_nu = minimum(se.pseudodiff_scaling)
    rho = se.rho_uppermantle

    lambda(k) = (rho * g + D * k^4) / (2 * eta * k * s_nu)
    k_dc = Array(domain.pseudodiff)[1, 1]
    k_max = maximum(domain.pseudodiff)

    lambda_dc = lambda(k_dc)
    lambda_kmax = lambda(k_max)
    return (dc = lambda_dc, k_max = lambda_kmax, overall = max(lambda_dc, lambda_kmax))
end

"""
$(TYPEDSIGNATURES)

Combine [`spectral_radius_estimate`](@ref) on `sim`'s RHS with the real-axis
stability limits of `EulerIntegrator`/`BS3Integrator`/`Tsit5Integrator` to report the implied stable
`dt` for each built-in stepper, plus the cross-check
[`analytic_lambda_bound`](@ref). `t` defaults to the start of `sim`'s time
span.
"""
function stiffness_report(sim::Simulation; t = sim.timer.t_span[1], kwargs...)
    lambda_max = spectral_radius_estimate(
        simulation_rhs_probe(sim),
        sim.now.u,
        nothing,
        t;
        kwargs...,
    )
    bound = analytic_lambda_bound(sim)
    dt_stable(alg) = stability_limit(alg) / lambda_max
    return (
        lambda_max = lambda_max,
        analytic_bound = bound,
        dt_euler = dt_stable(EulerIntegrator),
        dt_bs3 = dt_stable(BS3Integrator),
        dt_tsit5 = dt_stable(Tsit5Integrator),
    )
end
