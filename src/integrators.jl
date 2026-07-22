"""
$(TYPEDSIGNATURES)

Define a self-contained, dependency-free time integrator for in-place ODEs.

Available subtypes:
- [`EulerIntegrator`](@ref)
- [`BS3Integrator`](@ref)
- [`Tsit5Integrator`](@ref)
- [`RKCIntegrator`](@ref)
"""
abstract type AbstractIntegrator end

"""
$(TYPEDSIGNATURES)

Common driver interface shared by every stepper's integrator state
(`TableauIntegratorState` for RKTableau-based methods, `RKCIntegratorState` for `RKCIntegrator`).

`solve_to!`/`solve_to_adaptive!`/`advance_with_output!`/`step!` dispatch on
this abstract type; only `perform_step!` and the small generic hooks
`stepper_order`, `fsal_carryover!`, `steplog_entry`, `maybe_reestimate!` have
per-concrete-integrator methods.
"""
abstract type AbstractIntegratorState end

# Every integrator stores its settings concretely in its own float type `T`.
# `BS3Integrator()` infers `T` from the defaults below (`Float32`, the package-wide
# default — cf. `Timer(t_span; T = Float32)`); `BS3Integrator{Float64}(reltol = 1e-8)`
# pins it. Either way `init_integrator` converts every setting to the simulation's
# own element type, so the stored type is a matter of precision, not of dispatch.
#
# The defaults deliberately name no type variable: `@kwdef` also generates the
# `BS3Integrator(; ...)` method that leaves `T` to be inferred, and a `T`-dependent
# default (`eps(T)`) would make that method throw `UndefVarError: T`. Hence the
# concrete `eps(Float32)` floor and the `Inf32` ceiling, which widens to `Inf` for
# any `T` and so means "unbounded" in every precision.

"""
    EulerIntegrator(dt)
    EulerIntegrator(; dt)

Fixed-step explicit Euler. Non-adaptive, so `dt` is *the* step size and is
required — there is no error estimate to adapt it with.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct EulerIntegrator{T<:AbstractFloat} <: AbstractIntegrator
    "the fixed step size"
    dt::T
end

"""
    BS3Integrator(; kwargs...)
    BS3Integrator{T}(; kwargs...)

Bogacki-Shampine 3(2) embedded pair (FSAL, adaptive). Third-order accurate
solution with a second-order embedded error estimate.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct BS3Integrator{T<:AbstractFloat} <: AbstractIntegrator
    "relative error tolerance of the adaptive controller"
    reltol::T = 1.0f-5
    "absolute error tolerance of the adaptive controller"
    abstol::T = 1.0f-6
    "lower bound on the adaptive step size"
    dt_min::T = eps(Float32)
    "upper bound on the adaptive step size"
    dt_max::T = Inf32
    "initial step size; the controller grows it from here, at most 10x per step"
    dt0::T = 1.0f-3
end
"""
    Tsit5Integrator(; kwargs...)
    Tsit5Integrator{T}(; kwargs...)

Tsitouras 5(4) embedded pair (FSAL, adaptive). Fifth-order accurate solution
with a fourth-order embedded error estimate.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct Tsit5Integrator{T<:AbstractFloat} <: AbstractIntegrator
    "relative error tolerance of the adaptive controller"
    reltol::T = 1.0f-5
    "absolute error tolerance of the adaptive controller"
    abstol::T = 1.0f-6
    "lower bound on the adaptive step size"
    dt_min::T = eps(Float32)
    "upper bound on the adaptive step size"
    dt_max::T = Inf32
    "initial step size; the controller grows it from here, at most 10x per step"
    dt0::T = 1.0f-3
end
"""
    RKCIntegrator(; kwargs...)
    RKCIntegrator{T}(; kwargs...)

Stabilised explicit Runge-Kutta-Chebyshev method (RKC2, Sommeijer-Shampine-
Verwer 1997), second order, damped. Real-axis stability interval grows with the
*square* of the stage count. For stiff systems with ~1.75x speedup.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RKCIntegrator{T<:AbstractFloat} <: AbstractIntegrator
    "SSV damping parameter `ε`; must be `> 0`"
    damping::T = 2.0f0 / 13
    "safety factor for `dt * λ_max` when choosing the stage count"
    safety::T = 1.2f0
    "hard cap on the stage count per step"
    smax::Int = 200
    "re-run the power iteration every N accepted steps (`0` = never)"
    reestimate_every::Int = 0
    "relative error tolerance of the adaptive controller"
    reltol::T = 1.0f-5
    "absolute error tolerance of the adaptive controller"
    abstol::T = 1.0f-6
    "lower bound on the adaptive step size"
    dt_min::T = eps(Float32)
    "upper bound on the adaptive step size"
    dt_max::T = Inf32
    "initial step size; the controller grows it from here, at most 10x per step"
    dt0::T = 1.0f-3
end
isadaptive(::AbstractIntegrator) = true
isadaptive(::EulerIntegrator) = false

# --- settings an integrator hands to `init_integrator` ------------------------
#
# Every stepper setting lives on the integrator itself, so `init_integrator` and
# `integrate` take none of them as keywords: there is exactly one place a
# tolerance or step bound can come from.

integ_reltol(alg::AbstractIntegrator) = alg.reltol
integ_abstol(alg::AbstractIntegrator) = alg.abstol
integ_dt0(alg::AbstractIntegrator) = alg.dt0
integ_dt_min(alg::AbstractIntegrator) = alg.dt_min
integ_dt_max(alg::AbstractIntegrator) = alg.dt_max

# `EulerIntegrator` is non-adaptive: it runs no error control (the tolerances
# below are never read by `solve_to_fixed!`), and its fixed `dt` is the initial
# step, the floor and the ceiling at once.
integ_reltol(alg::EulerIntegrator) = one(alg.dt)
integ_abstol(alg::EulerIntegrator) = one(alg.dt)
integ_dt0(alg::EulerIntegrator) = alg.dt
integ_dt_min(alg::EulerIntegrator) = alg.dt
integ_dt_max(alg::EulerIntegrator) = alg.dt

# `alg`'s step-size settings, converted to the element type the integration
# actually runs in. The controller grows `dt0` from below (at most 10x per step,
# which is robust for stiff starts and avoids initial blow-ups), so a small
# initial step costs only a handful of extra steps.
function step_bounds(::Type{T}, alg) where {T}
    dtmin = T(integ_dt_min(alg))
    dtmax = T(integ_dt_max(alg))
    return clamp(T(integ_dt0(alg)), dtmin, dtmax), dtmin, dtmax
end

"""
$(TYPEDSIGNATURES)

The fixed step size of a non-adaptive integrator, i.e. of an
[`EulerIntegrator`](@ref). The semi-implicit mantle update in
[`update_dudt!`](@ref) (`ViscousMantle` on a laterally constant or rigid
lithosphere) discretises time itself, so it needs the step size up front —
which an adaptive controller cannot supply.
"""
fixed_dt(alg::EulerIntegrator) = alg.dt
fixed_dt(alg::AbstractIntegrator) = error(
    "$(nameof(typeof(alg))) is adaptive and has no fixed step size, but the " *
    "semi-implicit mantle update needs one. Use " *
    "`SolverOptions(integ = EulerIntegrator(dt = ...))`, or pick a mantle " *
    "rheology that integrates explicitly.",
)

# -----------------------------------------------------------------------------
# Butcher tableaus
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Butcher tableau for an (embedded) explicit Runge-Kutta method.

# Fields
- `A`: `s×s` strictly-lower-triangular stage-coefficient matrix.
- `c`: `s` node vector (`c[1] == 0`).
- `b`: `s` weights of the propagated (higher-order) solution.
- `btilde`: `s` weights of the *error estimate* (`b - bhat`); empty if method non-adaptive.
- `order`: order of the propagated solution (used by the step controller).
- `fsal`: whether the method is First-Same-As-Last (last stage of an
          accepted step equals the first stage of the next one).
"""
struct RKTableau{T}
    A::Matrix{T}
    c::Vector{T}
    b::Vector{T}
    btilde::Vector{T}
    order::Int
    fsal::Bool
end

nstages(tab::RKTableau) = length(tab.c)

function tableau(::EulerIntegrator, ::Type{T}) where {T}
    return RKTableau{T}(zeros(T, 1, 1), T[0], T[1], T[], 1, false)
end

function tableau(::BS3Integrator, ::Type{T}) where {T}
    A = zeros(T, 4, 4)
    A[2, 1] = 1 // 2
    A[3, 2] = 3 // 4
    A[4, 1] = 2 // 9
    A[4, 2] = 1 // 3
    A[4, 3] = 4 // 9
    c = T[0, 1//2, 3//4, 1]
    b = T[2//9, 1//3, 4//9, 0]                       # 3rd order
    bhat = T[7//24, 1//4, 1//3, 1//8]                # 2nd order
    btilde = b .- bhat
    return RKTableau{T}(A, c, b, btilde, 3, true)
end

function tableau(::Tsit5Integrator, ::Type{T}) where {T}
    # Tsitouras (2011) 5(4) pair, as used by OrdinaryDiffEq's `Tsit5`.
    c = T[0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]

    A = zeros(T, 7, 7)
    A[2, 1] = 0.161
    A[3, 1] = -0.008480655492356989
    A[3, 2] = 0.335480655492357
    A[4, 1] = 2.8971530571054935
    A[4, 2] = -6.359448489975075
    A[4, 3] = 4.3622954328695815
    A[5, 1] = 5.325864828439257
    A[5, 2] = -11.748883564062828
    A[5, 3] = 7.4955393428898365
    A[5, 4] = -0.09249506636175525
    A[6, 1] = 5.86145544294642
    A[6, 2] = -12.92096931784711
    A[6, 3] = 8.159367898576159
    A[6, 4] = -0.071584973281401
    A[6, 5] = -0.028269050394068383
    A[7, 1] = 0.09646076681806523
    A[7, 2] = 0.01
    A[7, 3] = 0.4798896504144996
    A[7, 4] = 1.379008574103742
    A[7, 5] = -3.290069515436081
    A[7, 6] = 2.324710524099774

    b = T[A[7, 1], A[7, 2], A[7, 3], A[7, 4], A[7, 5], A[7, 6], 0.0]  # FSAL

    btilde = T[
        -0.00178001105222577714,
        -0.0008164344596567469,
        0.007880878010261995,
        -0.1447110071732629,
        0.5823571654525552,
        -0.45808210592918697,
        0.015151515151515152,
    ]
    return RKTableau{T}(A, c, b, btilde, 5, true)
end

# -----------------------------------------------------------------------------
# Integrator state
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Running state and pre-allocated work arrays of an in-flight integration driven by
a tableau-based [`AbstractIntegrator`](@ref) (`EulerIntegrator`, `BS3Integrator`,
`Tsit5Integrator`). Analogous to a SciML *integrator* object — the algorithm
itself is the `alg` field — with `p` the user parameter object passed to the RHS
`f!(du, u, p, t)`. Built by [`init_integrator`](@ref); `RKCIntegrator` uses
[`RKCIntegratorState`](@ref) instead.
"""
mutable struct TableauIntegratorState{A,T,F,P,Alg<:AbstractIntegrator} <:
               AbstractIntegratorState
    f!::F
    p::P
    alg::Alg
    tableau::RKTableau{T}
    t::T
    dt::T
    u::A                 # current solution (== uprev during a step)
    unew::A              # candidate solution of the current step
    utmp::A              # stage temporary
    atmp::A              # error-estimate / scaling temporary
    ks::Vector{A}        # stage derivatives; ks[1] is the FSAL derivative
    reltol::T
    abstol::T
    dtmin::T
    dtmax::T
    facold::T            # PI-controller memory (previous accepted error)
    naccept::Int
    nreject::Int
    nf::Int              # number of RHS evaluations
end

"""
    init_integrator(f!, u0, tspan, alg::AbstractIntegrator, p = nothing)

Build (and prime) the integrator state for the in-place ODE `f!(du, u, p, t)`,
ready to be advanced by `solve_to!` / `step!`. Returns a
[`TableauIntegratorState`](@ref), or an [`RKCIntegratorState`](@ref) when `alg`
is an `RKCIntegrator`.

Tolerances and step-size bounds are read off `alg` — set them there, e.g.
`Tsit5Integrator(reltol = 1e-8)` or `EulerIntegrator(dt = 100.0)`.

See [`init_integrator(sim::Simulation)`](@ref) for the `Simulation`-level entry
point, which takes `alg` from `sim.opts.integ`.
"""
function init_integrator(f!, u0::A, tspan, alg::AbstractIntegrator, p = nothing) where {A}
    T = eltype(u0)
    tab = tableau(alg, T)
    s = nstages(tab)

    t0 = T(tspan[1])
    dt, dtmn, dtmx = step_bounds(T, alg)

    u = copy(u0)
    ks = [similar(u0) for _ = 1:s]

    integ = TableauIntegratorState(
        f!,
        p,
        alg,
        tab,
        t0,
        dt,
        u,
        similar(u0),
        similar(u0),
        similar(u0),
        ks,
        T(integ_reltol(alg)),
        T(integ_abstol(alg)),
        dtmn,
        dtmx,
        T(1e-4),
        0,
        0,
        0,
    )

    # Prime the FSAL first stage: ks[1] = f(t0, u0).
    f!(ks[1], u, p, t0)
    integ.nf += 1
    return integ
end

# -----------------------------------------------------------------------------
# Single step
# -----------------------------------------------------------------------------

# Compute the stages, the candidate solution `unew` and the scaled RMS error
# norm for a proposed step of size `dt`. `ks[1]` is assumed to already hold
# f(t, u) (FSAL / primed at init). The solution `u` and time `t` are left
# untouched; the driver decides whether to accept.
function perform_step!(integ::TableauIntegratorState, dt)
    tab = integ.tableau
    s = nstages(tab)
    t, u, p = integ.t, integ.u, integ.p
    ks, utmp, unew, atmp = integ.ks, integ.utmp, integ.unew, integ.atmp

    @inbounds for i = 2:s
        copyto!(utmp, u)
        for j = 1:(i-1)
            a = tab.A[i, j]
            iszero(a) && continue
            @. utmp += dt * a * ks[j]
        end
        integ.f!(ks[i], utmp, p, t + tab.c[i] * dt)
        integ.nf += 1
    end

    copyto!(unew, u)
    @inbounds for i = 1:s
        b = tab.b[i]
        iszero(b) && continue
        @. unew += dt * b * ks[i]
    end

    isempty(tab.btilde) && return zero(eltype(u))   # non-adaptive

    fill!(atmp, zero(eltype(atmp)))
    @inbounds for i = 1:s
        bt = tab.btilde[i]
        iszero(bt) && continue
        @. atmp += dt * bt * ks[i]
    end
    # Scale by abstol + reltol*max(|uprev|,|unew|); RMS norm as in OrdinaryDiffEq.
    @. atmp = atmp / (integ.abstol + integ.reltol * max(abs(u), abs(unew)))
    return norm(atmp) / sqrt(length(unew))
end

# -----------------------------------------------------------------------------
# Generic per-integrator hooks (small dispatch points used by the shared
# `solve_to_adaptive!`/controller code below, specialised per concrete
# `AbstractIntegratorState`; see `RKCIntegratorState`'s methods further down).
# -----------------------------------------------------------------------------

# Order of the *propagated* solution, used by the PI controller's exponent
# (same convention as the tableaus: the higher, returned order, not the
# embedded error estimator's order).
stepper_order(integ::TableauIntegratorState) = integ.tableau.order

# FSAL carry-over: reuse the last stage's derivative as next step's first
# stage, when the tableau supports it. No-op for non-FSAL methods.
fsal_carryover!(integ::TableauIntegratorState) =
    (integ.tableau.fsal && copyto!(integ.ks[1], integ.ks[end]); nothing)

# `steplog` entry for one accepted step. Tableau methods log `(t, dt)`;
# `RKCIntegratorState` widens this to `(t, dt, s)` (roadmap §5) without changing
# this method.
steplog_entry(integ::TableauIntegratorState, dt) = (integ.t, dt)

# Optional periodic spectral-radius re-estimation hook (roadmap §5); a no-op
# for tableau methods, which have no spectral-radius state.
maybe_reestimate!(::TableauIntegratorState) = nothing

# Element type of one `steplog_entry(integ, dt)` for `alg`, keyed on the
# *algorithm* rather than the integrator: `ForwardRecord` (src/inverse/recording.jl)
# needs this to size its step-log buffers before any integrator exists. Must be
# kept in sync with `steplog_entry`'s per-integrator-type return tuple above.
steplog_entry_type(::AbstractIntegrator, ::Type{T}) where {T} = Tuple{T,T}

# -----------------------------------------------------------------------------
# Step-size controller (Hairer's PI controller, cf. dopri5)
# -----------------------------------------------------------------------------

const _SAFE = 0.9
const _BETA = 0.04
const _FACMIN = 0.2      # smallest allowed dtnew/dt
const _FACMAX = 10.0     # largest allowed dtnew/dt

function controller_accept_dt(integ::AbstractIntegratorState, err, dt)
    order = stepper_order(integ)
    expo1 = 1 / order - _BETA * 0.75
    err = max(err, 1e-10)
    fac11 = err^expo1
    fac = fac11 / integ.facold^_BETA
    fac = max(1 / _FACMAX, min(1 / _FACMIN, fac / _SAFE))
    return dt / fac
end

function controller_reject_dt(integ::AbstractIntegratorState, err, dt)
    order = stepper_order(integ)
    expo1 = 1 / order - _BETA * 0.75
    fac11 = err^expo1
    return dt / min(1 / _FACMIN, fac11 / _SAFE)
end

# -----------------------------------------------------------------------------
# Drive the integrator up to `target` (a save/stop time)
# -----------------------------------------------------------------------------

# `steplog`, when a `Vector{<:Tuple}`, receives one `(t, dt)` entry per *accepted*
# step (`t` = time before the step, exactly as passed to the RHS) — the frozen
# dt-sequence the Phase-5 adjoint replays; `nothing` (default) records nothing.
function solve_to!(
    integ::AbstractIntegratorState,
    target,
    maxiters,
    steplog = nothing,
    progress = nothing,
)
    isadaptive(integ.alg) ?
    solve_to_adaptive!(integ, target, maxiters, steplog, progress) :
    solve_to_fixed!(integ, target, maxiters, steplog, progress)
end

function solve_to_adaptive!(
    integ::AbstractIntegratorState,
    target,
    maxiters,
    steplog = nothing,
    progress = nothing,
)
    T = typeof(integ.t)
    target = T(target)
    iters = 0
    while integ.t < target
        iters += 1
        iters > maxiters &&
            error("TableauIntegratorState: exceeded maxiters=$maxiters at t=$(integ.t)")

        dt = min(integ.dt, target - integ.t)
        clipped = dt < integ.dt
        err = perform_step!(integ, dt)

        if err <= 1 || dt <= integ.dtmin
            # accept
            steplog === nothing || push!(steplog, steplog_entry(integ, dt))
            integ.t += dt
            copyto!(integ.u, integ.unew)
            fsal_carryover!(integ)
            integ.naccept += 1
            # Only let a *natural* (unclipped) step drive the controller, so
            # clipping onto a save time does not shrink the running dt.
            if !clipped
                integ.dt = clamp(
                    controller_accept_dt(integ, err, dt),
                    integ.dtmin,
                    integ.dtmax,
                )
                integ.facold = max(err, 1e-4)
            end
            maybe_reestimate!(integ)
            report_progress!(progress, integ)
        else
            # reject: shrink and retry (u and t unchanged, so ks[1] stays valid)
            integ.nreject += 1
            integ.dt = max(controller_reject_dt(integ, err, dt), integ.dtmin)
        end
    end
    return integ
end

function solve_to_fixed!(
    integ::TableauIntegratorState,
    target,
    maxiters,
    steplog = nothing,
    progress = nothing,
)
    T = typeof(integ.t)
    target = T(target)
    iters = 0
    while integ.t < target
        iters += 1
        iters > maxiters &&
            error("TableauIntegratorState: exceeded maxiters=$maxiters at t=$(integ.t)")

        dt = min(integ.dt, target - integ.t)
        perform_step!(integ, dt)
        steplog === nothing || push!(steplog, (integ.t, dt))
        integ.t += dt
        copyto!(integ.u, integ.unew)
        # Euler is not FSAL: refresh the first stage for the next step.
        integ.f!(integ.ks[1], integ.u, integ.p, integ.t)
        integ.nf += 1
        report_progress!(progress, integ)
    end
    return integ
end

# =============================================================================
# RKCIntegrator: stabilised explicit Runge-Kutta-Chebyshev (RKC2, damped, s stages)
#
# Not a Butcher tableau: the stage count `s` is chosen per step from `dt` and
# an estimated spectral radius, and the update is a three-term Chebyshev
# recurrence rather than a fixed stage matrix. See `RKCIntegrator`'s docstring and
# roadmap `stabilise_dt.md` §2.2/§5.
# =============================================================================

# -----------------------------------------------------------------------------
# Chebyshev polynomial machinery (closed-form three-term recurrence, no
# tabulated coefficients).
# -----------------------------------------------------------------------------

# T_j(x), T_j'(x), T_j''(x) for every degree j = 0..s at a single point x, via
# the recurrence and its first two derivatives (obtained by differentiating
# the defining recurrence T_j = 2x T_{j-1} - T_{j-2} term by term). Returns
# three length-(s+1) vectors indexed `[j+1] == degree j`. Requires `s >= 1`.
function chebyshev_table(s::Int, x::T) where {T}
    Tv = Vector{T}(undef, s + 1)
    Dv = Vector{T}(undef, s + 1)
    Pv = Vector{T}(undef, s + 1)
    Tv[1], Dv[1], Pv[1] = one(T), zero(T), zero(T)
    Tv[2], Dv[2], Pv[2] = x, one(T), zero(T)
    @inbounds for j = 2:s
        Tv[j+1] = 2x * Tv[j] - Tv[j-1]
        Dv[j+1] = 2 * Tv[j] + 2x * Dv[j] - Dv[j-1]
        Pv[j+1] = 4 * Dv[j] + 2x * Pv[j] - Pv[j-1]
    end
    return Tv, Dv, Pv
end

# Scalar-test-equation evaluation of the *actual* s-stage recurrence (not a
# closed-form shortcut — the b_j/gammatilde 2nd-order correction means the
# method's stability function is no longer simply T_s(w0+w1 z)/T_s(w0), unlike
# the plain/undamped Chebyshev method): returns Y_s/Y_0 for `u' = z/dt * u`,
# `dt = 1`. Used by `rkc_stability_boundary`.
function rkc_scalar_map(mu, nu, mutilde, gammatilde, s::Int, z::T) where {T}
    F0 = z
    y0 = one(T)
    y1 = y0 + mutilde[1] * F0
    y2 = y0
    @inbounds for j = 2:s
        Fjm1 = z * y1
        ynext =
            mu[j] * y1 +
            nu[j] * y2 +
            (1 - mu[j] - nu[j]) * y0 +
            mutilde[j] * Fjm1 +
            gammatilde[j] * F0
        y2, y1 = y1, ynext
    end
    return y1
end

"""
$(TYPEDSIGNATURES)

Exact real-axis stability boundary of the `s`-stage damped RKC recurrence: the
largest `β > 0` such that `|Y_s/Y_0| <= 1` (evaluated by directly iterating
the actual recurrence via `rkc_scalar_map`, not a closed-form shortcut) for
`z ∈ [-β, 0]`. Found by bisection (mirrors `real_axis_stability_limit` for
`RKTableau`, `src/stability.jl`). Used both by `rkc_choose_stages` and as a
standalone regression check that the method's stage-vs-stability-boundary
scaling matches the literature (roadmap §2.2: `β(s) ≈ 0.65 s²` for the SSV
default damping — confirmed numerically to 3 significant figures for
`s` up to 200).
"""
function rkc_stability_boundary(s::Int, damping::T; tol = sqrt(eps(T))) where {T}
    mu, nu, mutilde, gammatilde, _ = rkc_coeffs(T, s, damping)
    stable(beta) =
        abs(rkc_scalar_map(mu, nu, mutilde, gammatilde, s, -beta)) <= 1 + sqrt(eps(T))
    lo, hi = zero(T), T(4 * s^2)
    while stable(hi)
        hi *= 2
    end
    while hi - lo > tol * max(one(T), hi)
        mid = (lo + hi) / 2
        stable(mid) ? (lo = mid) : (hi = mid)
    end
    return lo
end

"""
$(TYPEDSIGNATURES)

Per-stage recurrence coefficients `(mu, nu, mutilde, gammatilde, c)` for the
`s`-stage damped, second-order RKC method (Sommeijer-Shampine-Verwer 1997),
each a length-`s` vector indexed by stage `j` (`mu[1]`/`nu[1]`/`gammatilde[1]`
are unused — stage 1 only uses `mutilde[1]`). Cross-checked against the
production implementation in SUNDIALS' `LSRKStep` (`arkode_lsrkstep.c`,
`lsrkStep_TakeStepRKC`), since the roadmap author's own recollection of these
formulas from the SSV paper proved unreliable in two earlier implementation
attempts (both empirically falsified — see git history) before this one was
sourced and verified.

    w0 = 1 + damping/s²
    w1 = sinh(sφ) (w0²-1) / (s√(w0²-1) cosh(sφ) - w0 sinh(sφ)),  φ = acosh(w0)
    b_j = T_j''(w0) / T_j'(w0)²  for j >= 2, with the regularised base case
        b_0 = b_1 = 1/(2 w0)²  (T_1''/T_1'² is 0/1, degenerate, so b_1 is
        pinned by this closed form instead, *not* set equal to b_2)
    a_{j-1} = 1 - b_{j-1} T_{j-1}(w0)

    mu_j = 2 b_j w0 / b_{j-1},  nu_j = -b_j / b_{j-2}
    mutilde_j = 2 b_j w1 / b_{j-1},  gammatilde_j = -a_{j-1} mutilde_j
    mutilde_1 = b_1 w1

    Y_j = mu_j Y_{j-1} + nu_j Y_{j-2} + (1-mu_j-nu_j) Y_0
        + mutilde_j dt f(Y_{j-1}) + gammatilde_j dt f(Y_0),  j = 2,...,s
    Y_1 = Y_0 + mutilde_1 dt f(Y_0)

Verified (regression-tested in `test/test_integrators.jl`) to reproduce, on the
scalar test equation `u' = λu` (`z = λ dt`): `Y_s/Y_0 = 1 + z + z²/2 + O(z³)`
exactly (second-order consistency, to floating-point precision, for every `s`
tested) and a real-axis stability boundary `β(s)` with `β(s)/s² → 0.653` as
`s → ∞` for the SSV default damping `2/13` — matching the literature value
this roadmap's design discussion anticipated (§1/§2.2).

`c` are companion stage-time fractions (`t + c[j-1]*dt` is the evaluation time
of stage `j`'s RHS call), obtained by applying the *same* linear recurrence to
the scalar sequence generated by `y' = 1, y(0) = 0` (exact solution `y(t) =
t`) — a standard, self-consistent way to derive quadrature nodes for a
recursion of this shape, guaranteeing the stage times are consistent with the
recurrence's own weights for RHS's with explicit time dependence (e.g. ramped
loads).
"""
function rkc_coeffs(::Type{T}, s::Int, damping) where {T}
    s >= 2 ||
        throw(ArgumentError("RKCIntegrator requires at least 2 stages (got s=$s)"))
    damping > 0 || throw(
        ArgumentError(
            "RKCIntegrator requires damping > 0 (w1's closed form is singular at the undamped limit w0=1)",
        ),
    )
    w0 = one(T) + T(damping) / s^2
    Tv, Dv, Pv = chebyshev_table(s, w0)     # Tv[j+1] == T_j(w0), etc.

    temp1 = w0^2 - one(T)
    temp2 = sqrt(temp1)
    arg = s * log(w0 + temp2)
    w1 = sinh(arg) * temp1 / (cosh(arg) * s * temp2 - w0 * sinh(arg))

    mu = zeros(T, s)
    nu = zeros(T, s)
    mutilde = zeros(T, s)
    gammatilde = zeros(T, s)
    c = zeros(T, s)

    b1 = one(T) / (2 * w0)^2               # b_0 = b_1, closed form (regularises T_1''/T_1'²)
    mutilde[1] = b1 * w1
    c[1] = mutilde[1]

    bjm2, bjm1 = b1, b1
    @inbounds for j = 2:s
        bj = Pv[j+1] / Dv[j+1]^2
        mu[j] = 2 * bj * w0 / bjm1
        nu[j] = -bj / bjm2
        mutilde[j] = 2 * bj * w1 / bjm1
        a_jm1 = one(T) - bjm1 * Tv[j]       # Tv[j] == T_{j-1}(w0)
        gammatilde[j] = -a_jm1 * mutilde[j]

        cjm2 = j == 2 ? zero(T) : c[j-2]
        c[j] = mu[j] * c[j-1] + nu[j] * cjm2 + mutilde[j] + gammatilde[j]

        bjm2, bjm1 = bjm1, bj
    end

    return mu, nu, mutilde, gammatilde, c
end

# Smallest stage count (clamped to [2, smax]) whose exact stability boundary
# covers `safety * dt * lambda_max`, seeded by the closed-form asymptotic
# `β(s) ≈ 0.65 s²` (roadmap §2.2/§5) and refined against the exact boundary
# (`rkc_stability_boundary`) so the result is correct regardless of how
# accurate that seed constant is — a bad seed only costs a few extra integer
# increments here, utterly negligible next to the RHS evaluations the chosen
# `s` will cost. If `smax` is reached and still insufficient, `s = smax` is
# returned anyway and the ordinary error-based reject/shrink cycle (not a
# special code path here) drives `dt` down on retry.
function rkc_choose_stages(
    dt,
    lambda_max::T,
    damping::T,
    smax::Int;
    safety::T = T(1.2),
) where {T}
    z = safety * dt * lambda_max
    z <= 0 && return 2
    s = clamp(ceil(Int, sqrt(z / T(0.65))), 2, smax)
    while s < smax && rkc_stability_boundary(s, damping) < z
        s += 1
    end
    return s
end

# Wrap `f!` to count its invocations into a `Ref{Int}`, used to fold the RHS
# evaluations spent on spectral-radius (re-)estimation honestly into `nf`.
function _counting_wrapper(f!)
    n = Ref(0)
    counted! = (du, u, p, t) -> (f!(du, u, p, t); n[] += 1; nothing)
    return counted!, n
end

# -----------------------------------------------------------------------------
# RKCIntegrator integrator state
# -----------------------------------------------------------------------------

"""
    RKCIntegratorState

Mutable state and O(1) (independent of stage count) work arrays for `RKCIntegrator`.
The rolling Chebyshev recurrence needs only three grid-sized buffers for the
`Y_{j-2}, Y_{j-1}, Y_j` sequence (`ym2`, `ym1`, `unew`, cycled by reference
swap — no per-step allocation) plus `F0` (the RHS at `Y_0`, constant through a
step) and `Fj` (the RHS at the current stage). The per-step coefficient
vectors from `rkc_coeffs` are small (`O(s)` scalars, `s <= smax`, a few KB at
most) and are *not* part of this O(1)-work-array guarantee, which concerns the
grid-sized state only.
"""
mutable struct RKCIntegratorState{A,T,F,P} <: AbstractIntegratorState
    f!::F
    p::P
    alg::RKCIntegrator
    t::T
    dt::T
    u::A                 # current solution (== uprev during a step)
    unew::A              # candidate solution of the current step
    ym1::A                # Y_{j-1} rolling buffer
    ym2::A                # Y_{j-2} rolling buffer
    F0::A                 # f(Y_0), constant through a step
    Fj::A                 # f(Y_{j-1}), refreshed every stage
    atmp::A               # error-estimate / scaling temporary
    lambda_max::T          # cached spectral-radius estimate
    s::Int                 # stage count used by the most recent step
    reltol::T
    abstol::T
    dtmin::T
    dtmax::T
    facold::T
    naccept::Int
    nreject::Int
    nf::Int
end

# `lambda_maxiter`/`lambda_tol` stay keywords: they tune the power iteration that
# seeds the spectral-radius estimate, not the RKC method itself.
function init_integrator(
    f!,
    u0::A,
    tspan,
    alg::RKCIntegrator,
    p = nothing;
    lambda_maxiter::Int = 100,
    lambda_tol = 1e-2,
) where {A}
    T = eltype(u0)

    t0 = T(tspan[1])
    dt, dtmn, dtmx = step_bounds(T, alg)

    probe = p isa Simulation ? snapshotting_probe(f!, p) : f!
    counted_f!, nf0 = _counting_wrapper(probe)
    lambda_max = spectral_radius_estimate(
        counted_f!,
        u0,
        p,
        t0;
        maxiter = lambda_maxiter,
        tol = T(lambda_tol),
    )

    return RKCIntegratorState(
        f!,
        p,
        alg,
        t0,
        dt,
        copy(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        T(lambda_max),
        0,
        T(integ_reltol(alg)),
        T(integ_abstol(alg)),
        dtmn,
        dtmx,
        T(1e-4),
        0,
        0,
        nf0[],
    )
end

# -----------------------------------------------------------------------------
# RKCIntegrator single step
# -----------------------------------------------------------------------------

# Same contract as the tableau `perform_step!`: `u`/`t` are left untouched, the
# candidate solution is left in `integ.unew`, and the scaled RMS error norm is
# returned; the driver decides whether to accept.
function perform_step!(integ::RKCIntegratorState, dt)
    alg = integ.alg
    p, t = integ.p, integ.t
    u = integ.u
    T = eltype(u)

    s = rkc_choose_stages(
        T(dt),
        integ.lambda_max,
        T(alg.damping),
        alg.smax;
        safety = T(alg.safety),
    )
    integ.s = s
    mu, nu, mutilde, gammatilde, c = rkc_coeffs(T, s, T(alg.damping))

    F0, Fj = integ.F0, integ.Fj
    integ.f!(F0, u, p, t)
    integ.nf += 1

    y0 = u                                             # Y_0, never mutated below
    y1, y2, ynext = integ.ym1, integ.ym2, integ.unew    # rolling Y_{j-1}, Y_{j-2}, scratch
    @. y1 = y0 + mutilde[1] * dt * F0                   # Y_1
    copyto!(y2, y0)                                     # Y_0 snapshot, rolled as "Y_{j-2}" from j=2

    @inbounds for j = 2:s
        integ.f!(Fj, y1, p, t + c[j-1] * dt)
        integ.nf += 1
        muj, nuj, mtj, gtj = mu[j], nu[j], mutilde[j], gammatilde[j]
        @. ynext =
            muj * y1 + nuj * y2 + (1 - muj - nuj) * y0 + mtj * dt * Fj + gtj * dt * F0
        y2, y1, ynext = y1, ynext, y2
    end
    # Y_s now lives in `y1` (the final rotation moved the last-computed stage there).

    # SSV/RKC embedded error estimate (Sommeijer–Shampine–Verwer 1997, §4), the
    # exact form used by SUNDIALS' `LSRKStep` (`arkode_lsrkstep.c`,
    # `lsrkStep_TakeStepRKC`, constants `p8 = 0.8`, `p4 = 0.4`):
    #
    #   Est = 0.8·(y0 − Y_s) + 0.4·dt·(F0 + f(Y_s))
    #
    # On the scalar test equation `u' = λu` (`z = λ dt`) this is `Est/y0 =
    # 0.8(1−R(z)) + 0.4 z (1+R(z))`, which is O(z³) as z→0 (it genuinely tracks
    # the 2nd-order method's O(dt³) local truncation error), and stays *bounded*
    # (≈0.4|z|) as z→−∞ instead of blowing up. That boundedness is the whole
    # point: the previous difference-from-Euler indicator, `Y_s − (y0 + dt F0)`,
    # measured how far the stable RKC step diverges from an *unstable* Euler
    # predictor, so it exploded like the Euler stability defect (~|1+z|) for
    # every mode with z ≲ −2 and pinned dt at the Euler stability limit — which
    # made `rkc_choose_stages` never pick more than s = 2 stages and defeated
    # the entire √stiffness advantage (roadmap §6 performance box).
    #
    # Cost: one extra RHS eval per step — `f(Y_s)` — which the original
    # "zero-extra-eval" design skipped. That was a false economy: without a
    # usable estimate the controller took ~10–100× more (tiny) steps. The eval
    # is not reused as the next step's F0 (RKC is deliberately non-FSAL, roadmap
    # §2.3), so `nf` honestly counts it.
    integ.f!(Fj, y1, p, t + dt)
    integ.nf += 1
    atmp = integ.atmp
    p8, p4 = T(0.8), T(0.4)
    @. atmp =
        (p8 * (y0 - y1) + p4 * dt * (F0 + Fj)) /
        (integ.abstol + integ.reltol * max(abs(y0), abs(y1)))
    err = norm(atmp) / sqrt(length(atmp))

    integ.unew, integ.ym1, integ.ym2 = y1, y2, ynext
    return err
end

# -----------------------------------------------------------------------------
# RKCIntegrator generic-hook methods
# -----------------------------------------------------------------------------

stepper_order(::RKCIntegratorState) = 2                # RKC2 is second order

fsal_carryover!(::RKCIntegratorState) = nothing         # not FSAL (roadmap §2.3)

# Widen the steplog tuple to include the stage count used, for a future frozen
# replay (roadmap §5/§7); EulerIntegrator/BS3Integrator/Tsit5Integrator logging is untouched (their
# `steplog_entry` method above still returns `(t, dt)`).
steplog_entry(integ::RKCIntegratorState, dt) = (integ.t, dt, integ.s)

# Matches the 3-tuple `steplog_entry` above; see `steplog_entry_type`'s
# definition (near the tableau-path `steplog_entry`) for why this is keyed on
# the algorithm rather than the integrator.
steplog_entry_type(::RKCIntegrator, ::Type{T}) where {T} = Tuple{T,T,Int}

function maybe_reestimate!(integ::RKCIntegratorState)
    n = integ.alg.reestimate_every
    (n > 0 && integ.naccept % n == 0) || return nothing
    probe = integ.p isa Simulation ? snapshotting_probe(integ.f!, integ.p) : integ.f!
    counted_f!, nf0 = _counting_wrapper(probe)
    integ.lambda_max = spectral_radius_estimate(counted_f!, integ.u, integ.p, integ.t)
    integ.nf += nf0[]
    return nothing
end

# -----------------------------------------------------------------------------
# RKCIntegrator frozen replay (discrete adjoint, roadmap §7)
# -----------------------------------------------------------------------------

# The frozen replay is expressed as ONE FLAT LOOP OVER STAGES, not as a loop over
# steps containing a loop over that step's stages. This is an AD-compile-time
# decision, and it is the whole reason `RKCStage`/`rkc_stage_plan` exist.
#
# Enzyme's reverse transform must tape every primal value that the forward
# overwrites. At loop depth 1 a taped value costs one flat, `n`-sized allocation
# indexed by the induction variable. At loop depth 2 with an inner bound that is
# *not* loop-invariant — and `s` genuinely varies from step to step, it is chosen
# per step by `rkc_choose_stages` — Enzyme cannot form a rectangular cache and
# falls back to a jagged two-level allocation, emitted *per taped value*. Since
# the RHS (`update_diagnostics!`) is large (FFT round-trips, ~15 grid broadcasts,
# a convolution, a data-dependent sparse-diagnostics branch) and the recurrence
# overwrites five grid-sized buffers per stage, a nested formulation multiplies
# that jagged apparatus across hundreds of taped values.
#
# Flattening moves every RHS eval to a single call site at depth 1, so the
# differentiated region holds ONE inlined copy of `update_diagnostics!` — fewer
# than the Euler replay's two — over a flat tape. The stage plan itself is built
# outside `Enzyme.autodiff` and passed as `Const`, which also hoists out the
# coefficient machinery (`rkc_coeffs`: `cosh`/`sinh`/`log`, per-degree Chebyshev
# tables, allocation) and the stage-time arithmetic `t + c[j-1]*dt`.

"""
    RKCStage{T}

One entry of a flat RKC replay plan — one *RHS evaluation* — as
`(t_eval, mu, nu, mutilde, gammatilde, dt, is_first)`. Every stage, including the
two special ones, is driven through the single general update

    Y_next = mu*Y_{j-1} + nu*Y_{j-2} + (1-mu-nu)*Y_0 + mutilde*dt*f(Y_{j-1}) + gammatilde*dt*f(Y_0)

so the differentiated loop body needs no per-stage branching beyond `is_first`
(see [`rkc_stage_plan`](@ref) for the coefficient choices that make this exact).
"""
const RKCStage{T} = Tuple{T,T,T,T,T,T,Bool}

"""
$(TYPEDSIGNATURES)

Flatten a recorded `(t, dt, s)` step sequence into a per-stage
[`RKCStage`](@ref) plan for [`rkc_replay_stages!`](@ref). Purely a function of
the recording and `damping` — state-independent, hence `Const` under AD.

Each frozen step contributes `s + 1` entries, matching `perform_step!`'s `s + 1`
RHS evaluations at `t, t + c₁dt, …, t + c_{s-1}dt, t + dt`, with the two special
stages folded into the general update by coefficient choice:

| stage             | `(mu, nu, mutilde, gammatilde)`          | reduces to                     |
|:------------------|:-----------------------------------------|:-------------------------------|
| `j = 1`           | `(0, 0, mutilde[1], 0)`                  | `Y_0 + mutilde[1]·dt·f(Y_0)`   |
| `j = 2…s`         | `(mu[j], nu[j], mutilde[j], gammatilde[j])` | the SSV recurrence          |
| error-estimate    | `(1, 0, 0, 0)`                           | `Y_s` (state unchanged)        |

The trailing entry replays `perform_step!`'s embedded-error RHS eval at
`(Y_s, t + dt)`. Its error *value* is discarded (replay never rejects), but the
eval itself is part of the forward's deterministic trajectory and must happen:
it applies the BC to `Y_s` in place and advances the sim's sparse-diagnostic
counter (`count_sparse_updates`, `src/simulation.jl`) exactly as the forward did.
Dropping it would desynchronise that counter and leave `Y_s` un-BC'd. Choosing
`(1, 0, 0, 0)` makes the update a no-op on the state, so the eval happens without
perturbing the recurrence.
"""
function rkc_stage_plan(::Type{T}, steps, damping) where {T}
    plan = RKCStage{T}[]
    d = T(damping)
    for (t, dt, s) in steps
        append_rkc_stages!(plan, T(t), T(dt), s, d)
    end
    return plan
end

function append_rkc_stages!(
    plan::Vector{RKCStage{T}},
    t::T,
    dt::T,
    s::Int,
    damping::T,
) where {T}
    mu, nu, mutilde, gammatilde, c = rkc_coeffs(T, s, damping)
    z, o = zero(T), one(T)
    push!(plan, (t, z, z, mutilde[1], z, dt, true))                 # Y_1
    @inbounds for j = 2:s
        push!(
            plan,
            (t + c[j-1] * dt, mu[j], nu[j], mutilde[j], gammatilde[j], dt, false),
        )
    end
    push!(plan, (t + dt, o, z, z, z, dt, false))                    # error-estimate eval
    return plan
end

"""
$(TYPEDSIGNATURES)

Replay a flat [`rkc_stage_plan`](@ref) in place. `y1` must hold the running state
on entry (the interval-start state) and holds the accepted `Y_s` of the last step
on exit; `y0`, `y2`, `ynext`, `F`, `F0` are caller-owned scratch buffers, all
`similar(y1)`. `f!(du, x, p, t)` is the RHS (`update_diagnostics!` bound to a sim
via `p`).

Reproduces `perform_step!`'s recurrence bit for bit for the recorded
`(t, dt, s)` sequence — the stage count is read from the recording, never
re-selected from a spectral radius — so this is the primitive both the non-AD
`replay_interval!` (`src/inverse/recording.jl`) and the checkpointing extension's
reverse pass drive.

The buffers keep FIXED identities across the loop (roles advanced by `copyto!`,
not by swapping references): swapping array *pointers* creates SSA phi-nodes on
array types that blow up Enzyme's TypeTree analysis, and the two extra copies per
stage are negligible next to an RHS eval.
"""
function rkc_replay_stages!(f!, y0, y1, y2, ynext, F, F0, p, plan)
    @inbounds for m in eachindex(plan)
        te, muj, nuj, mtj, gtj, h, isfirst = plan[m]
        f!(F, y1, p, te)                    # the single RHS call site, loop depth 1
        # CAUTION: the RHS mutates its state argument in place (`apply_bc!(u, …)`
        # in `update_diagnostics!`), so Y_0 must be snapshotted *after* the eval —
        # `perform_step!` likewise reads `y0`/`y2` from the post-eval state. Taking
        # these copies before `f!` silently replays an un-BC'd Y_0 and the step
        # drifts from the recording at ~1e-13.
        if isfirst                          # new step: Y_0 ← the accepted Y_s
            copyto!(y0, y1)
            copyto!(y2, y1)
            copyto!(F0, F)                  # F0 = f(Y_0), constant through the step
        end
        @. ynext =
            muj * y1 + nuj * y2 + (1 - muj - nuj) * y0 + mtj * h * F + gtj * h * F0
        copyto!(y2, y1)                     # Y_{j-2} <- Y_{j-1}
        copyto!(y1, ynext)                  # Y_{j-1} <- Y_j
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

"""
    integrate(f!, u0, tspan, alg::AbstractIntegrator, p = nothing; kwargs...) -> (ts, us)

Integrate the in-place ODE `f!(du, u, p, t)` from `tspan[1]` to `tspan[2]`.

Returns the sorted save times `ts` and a vector `us` of solution snapshots
(copies) at those times.

Error tolerances and step-size bounds are fields of `alg` — e.g.
`integrate(f!, u0, tspan, Tsit5Integrator(reltol = 1e-8))` or
`integrate(f!, u0, tspan, EulerIntegrator(dt = 0.05))`.

Keyword arguments:
- `saveat`: times at which to store the solution (defaults to `tspan[2]`).
  Must lie within `tspan`. The endpoint is always included.
- `maxiters`: safety cap on the number of steps per save interval.
"""
function integrate(
    f!,
    u0,
    tspan,
    alg::AbstractIntegrator,
    p = nothing;
    saveat = nothing,
    maxiters = 10_000_000,
)
    T = eltype(u0)
    integ = init_integrator(f!, u0, tspan, alg, p)

    t0, tend = T(tspan[1]), T(tspan[2])
    save = saveat === nothing ? T[tend] : sort!(unique(T.(collect(saveat))))
    save[end] == tend || push!(save, tend)
    all(t0 .<= save .<= tend) || error("`saveat` times must lie within `tspan`.")

    us = typeof(integ.u)[]
    ts = T[]
    for tsave in save
        solve_to!(integ, tsave, maxiters)
        push!(ts, integ.t)
        push!(us, copy(integ.u))
    end
    return ts, us
end

# =============================================================================
# Integration with a `Simulation`.
#
# `run!`, `init_integrator` and `step!` drive a `Simulation` with the built-in
# stepper. They dispatch on the *abstract* `AbstractIntegrator`, so `BS3Integrator`/`Tsit5Integrator`/
# `EulerIntegrator` all share one path. Output is written by reusing `nc_affect!` and
# `nout_affect!`, which only read `integrator.p` (== the `Simulation`) and
# `integrator.t` — both of which `TableauIntegratorState` provides.
# =============================================================================

const STEPPER_MAXITERS = 10_000_000

# Build a primed integrator from a Simulation's options. `init_problem!` and
# `t_computation_0` must already have been set by the caller.
#
# The integrator carries its own tolerances and step bounds, so there is nothing
# to forward here: `init_integrator` reads them off `sim.opts.integ`.
build_integrator(sim) = init_integrator(
    update_diagnostics!,
    sim.now.u,
    sim.timer.t_span,
    sim.opts.integ,
    sim,
)

# Earliest pending recording time across all attached `SimulatedObservable`s
# (roadmap §4c item 2), or `nothing` if none are pending.
function _next_simobs_time(sim)
    t = nothing
    for so in sim.simobs
        tso = next_simobs_time(so)
        tso === nothing && continue
        t = t === nothing ? tso : min(t, tso)
    end
    return t
end

# Next pending output time across the native, netCDF and simulated-observable
# streams.
function _next_output_time(sim)
    tn =
        (length(sim.nout.t) >= 1 && sim.nout.k <= length(sim.nout.t)) ?
        sim.nout.t[sim.nout.k] : nothing
    tc =
        (length(sim.ncout.t) >= 1 && sim.ncout.k <= length(sim.ncout.t)) ?
        sim.ncout.t[sim.ncout.k] : nothing
    ts = _next_simobs_time(sim)
    t = tn === nothing ? tc : (tc === nothing ? tn : min(tn, tc))
    return t === nothing ? ts : (ts === nothing ? t : min(t, ts))
end

# Advance the integrator up to `target`, stopping exactly on every output time
# in between and writing output there.
function advance_with_output!(
    integ::AbstractIntegratorState,
    sim,
    target,
    maxiters = STEPPER_MAXITERS,
    progress = nothing,
)
    T = typeof(integ.t)
    target = T(target)
    while true
        te = _next_output_time(sim)
        if te === nothing || te > target
            solve_to!(integ, target, maxiters, nothing, progress)
            return integ
        end
        solve_to!(integ, te, maxiters, nothing, progress)
        # Fire netCDF first then native output (matches previous callback order),
        # then any simulated observables pending at this time.
        if length(sim.ncout.t) >= 1 &&
           sim.ncout.k <= length(sim.ncout.t) &&
           sim.ncout.t[sim.ncout.k] == te
            nc_affect!(integ, progress)
        end
        if length(sim.nout.t) >= 1 &&
           sim.nout.k <= length(sim.nout.t) &&
           sim.nout.t[sim.nout.k] == te
            nout_affect!(integ, progress)
        end
        for so in sim.simobs
            next_simobs_time(so) == te && record!(so, sim)
        end
    end
end

# `run!(sim::Simulation)` and `init_integrator(sim::Simulation)` — the two entry
# points whose *signature* mentions `Simulation` — live in simulation.jl, which
# is included after this file (see the note at that include in FastIsostasy.jl).

"""
$(TYPEDSIGNATURES)

Advance `integrator` by the interval `Δt`, using internal adaptive substeps and
writing any output that falls within the interval. `force_dt` is accepted for
backward compatibility; the step always stops exactly at `t + Δt`.
"""
function step!(integ::AbstractIntegratorState, Δt, force_dt::Bool = true)
    advance_with_output!(integ, integ.p, integ.t + Δt)
    return nothing
end
