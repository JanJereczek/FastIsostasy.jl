# =============================================================================
# Self-contained explicit Runge-Kutta integrators
#
# These are a lightweight, dependency-free alternative to the OrdinaryDiffEq.jl
# solvers used in `run!`. They are designed to plug into the same in-place RHS
# convention as SciML, i.e. `f!(du, u, p, t)`, and to work generically for any
# array type `u` (CPU `Matrix`, `CuArray`, ...) since every kernel is expressed
# as a broadcast.
#
# Implemented so far:
#   - `FIEuler`  : fixed-step explicit Euler (for the implicit/laterally-constant
#                  workflow that needs a fixed time step).
#   - `FIBS3`    : Bogacki-Shampine 3(2), FSAL, adaptive.
#   - `FITsit5`  : Tsitouras 5(4), FSAL, adaptive.
#
# The adaptive step-size controller and the scaled error norm mirror the
# defaults of OrdinaryDiffEq (Hairer's PI controller from `dopri5`, RMS error
# norm scaled by `abstol + reltol * max(|uprev|, |unew|)`), so that a given
# `reltol` produces a comparable number of steps across both backends.
# =============================================================================

abstract type FIAlgorithm end

# Common driver interface shared by every stepper's integrator state
# (`FIIntegrator` for RKTableau-based methods, `FIRKCIntegrator` for `FIRKC`).
# `solve_to!`/`solve_to_adaptive!`/`advance_with_output!`/`step!` dispatch on
# this abstract type; only `perform_step!` and the small generic hooks
# `stepper_order`, `fsal_carryover!`, `steplog_entry`, `maybe_reestimate!` have
# per-concrete-integrator methods.
abstract type AbstractFIIntegrator end

"""
    FIEuler()

Fixed-step explicit Euler. Non-adaptive: the step size is taken from `dt0`.
"""
struct FIEuler <: FIAlgorithm end

"""
    FIBS3()

Bogacki-Shampine 3(2) embedded pair (FSAL, adaptive). Third-order accurate
solution with a second-order embedded error estimate.
"""
struct FIBS3 <: FIAlgorithm end

"""
    FITsit5()

Tsitouras 5(4) embedded pair (FSAL, adaptive). Fifth-order accurate solution
with a fourth-order embedded error estimate.
"""
struct FITsit5 <: FIAlgorithm end

isadaptive(::FIAlgorithm) = true
isadaptive(::FIEuler) = false

"""
    FIRKC(; damping = 2/13, safety = 1.2, smax = 200, reestimate_every = 0)

Stabilised explicit Runge-Kutta-Chebyshev method (RKC2, Sommeijer-Shampine-
Verwer 1997), second order, damped. Unlike a classical RK tableau, its
real-axis stability interval grows with the *square* of the stage count
instead of being fixed per stage, so the number of RHS evaluations needed to
advance a stiff step scales with `sqrt(stiffness)` rather than `stiffness`
(roadmap `stabilise_dt.md` §2.2). All coefficients are generated at each step
from a closed-form three-term Chebyshev recurrence (no tabulated data, see
`rkc_coeffs`); the stage count is chosen from `dt` and an internally
estimated spectral radius, then the usual PI step-size controller adjusts
`dt` from a local error estimate — same conventions as `FIBS3`/`FITsit5`.
**Caveat:** the error estimate is a simplified embedded-Euler indicator (see
`perform_step!`), not SSV's own calibration — at a given `reltol` it is
measurably more conservative (lower global accuracy) than `FITsit5`; use a
visibly tighter `reltol` for comparable accuracy. This does not affect
stability, which comes from the independent stage-count selection.

- `damping`: SSV's damping parameter `ε`, trading a small reduction in the
  real-axis stability boundary for internal stability (robustness to
  non-normal/complex spectra). Must be `> 0` — the closed-form `w1` below has
  a removable singularity at the undamped limit `ε = 0` (`w0 = 1`) that is not
  handled specially; use a small positive value (the SSV default `2/13`) if
  in doubt.
- `safety`: multiplicative safety factor applied to `dt * λ_max` when picking
  the stage count (as in the original RKC/ROCK codes).
- `smax`: hard cap on the stage count per step. If the ideal stage count for a
  proposed `dt` would exceed this, the step is attempted at `smax` stages
  anyway; if that is genuinely insufficient the step's error estimate comes
  back large and the *existing* reject/shrink cycle (not a special code path)
  drives `dt` down until `smax` stages are enough.
- `reestimate_every`: re-run the spectral-radius power iteration every this
  many *accepted* steps (`0`, the default, disables re-estimation — the
  estimate from `init_fi` is assumed valid for the lifetime of the
  integration, correct as long as the coefficient fields defining the RHS's
  Jacobian are time-independent; see roadmap §5/§8).
"""
@kwdef struct FIRKC <: FIAlgorithm
    damping::Float64 = 2 / 13
    safety::Float64 = 1.2
    smax::Int = 200
    reestimate_every::Int = 0
end

# -----------------------------------------------------------------------------
# Butcher tableaus
# -----------------------------------------------------------------------------

"""
    RKTableau{T}

Butcher tableau for an (embedded) explicit Runge-Kutta method.

- `A`       : `s×s` strictly-lower-triangular stage-coefficient matrix.
- `c`       : `s` node vector (`c[1] == 0`).
- `b`       : `s` weights of the propagated (higher-order) solution.
- `btilde`  : `s` weights of the *error estimate* (`b - bhat`); empty if the
              method is non-adaptive.
- `order`   : order of the propagated solution (used by the step controller).
- `fsal`    : whether the method is First-Same-As-Last (last stage of an
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

function tableau(::FIEuler, ::Type{T}) where {T}
    return RKTableau{T}(zeros(T, 1, 1), T[0], T[1], T[], 1, false)
end

function tableau(::FIBS3, ::Type{T}) where {T}
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

function tableau(::FITsit5, ::Type{T}) where {T}
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
    FIIntegrator

Mutable state and pre-allocated work arrays for a [`FIAlgorithm`](@ref).
Analogous to a SciML integrator: `p` is the user parameter object passed to the
RHS `f!(du, u, p, t)`.
"""
mutable struct FIIntegrator{A, T, F, P, Alg <: FIAlgorithm} <: AbstractFIIntegrator
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

function init_fi(f!, u0::A, tspan, alg::FIAlgorithm, p = nothing;
    reltol = 1e-5,
    abstol = 1e-6,
    dt0 = nothing,
    dtmin = nothing,
    dtmax = nothing,
) where {A}
    T = eltype(u0)
    tab = tableau(alg, T)
    s = nstages(tab)

    t0, tend = T(tspan[1]), T(tspan[2])
    span = tend - t0

    # Default to a small initial step and let the controller grow it (capped at
    # 10x/step); this is robust for stiff-ish starts and avoids initial blow-ups.
    dt = dt0 === nothing ? abs(span) / 10_000 : T(dt0)
    dtmx = dtmax === nothing ? abs(span) : T(dtmax)
    dtmn = dtmin === nothing ? eps(T) * max(abs(t0), abs(tend)) : T(dtmin)
    dt = clamp(dt, dtmn, dtmx)

    u = copy(u0)
    ks = [similar(u0) for _ in 1:s]

    integ = FIIntegrator(
        f!, p, alg, tab,
        t0, dt,
        u, similar(u0), similar(u0), similar(u0), ks,
        T(reltol), T(abstol), dtmn, dtmx,
        T(1e-4), 0, 0, 0,
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
function perform_step!(integ::FIIntegrator, dt)
    tab = integ.tableau
    s = nstages(tab)
    t, u, p = integ.t, integ.u, integ.p
    ks, utmp, unew, atmp = integ.ks, integ.utmp, integ.unew, integ.atmp

    @inbounds for i in 2:s
        copyto!(utmp, u)
        for j in 1:i-1
            a = tab.A[i, j]
            iszero(a) && continue
            @. utmp += dt * a * ks[j]
        end
        integ.f!(ks[i], utmp, p, t + tab.c[i] * dt)
        integ.nf += 1
    end

    copyto!(unew, u)
    @inbounds for i in 1:s
        b = tab.b[i]
        iszero(b) && continue
        @. unew += dt * b * ks[i]
    end

    isempty(tab.btilde) && return zero(eltype(u))   # non-adaptive

    fill!(atmp, zero(eltype(atmp)))
    @inbounds for i in 1:s
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
# `AbstractFIIntegrator`; see `FIRKCIntegrator`'s methods further down).
# -----------------------------------------------------------------------------

# Order of the *propagated* solution, used by the PI controller's exponent
# (same convention as the tableaus: the higher, returned order, not the
# embedded error estimator's order).
stepper_order(integ::FIIntegrator) = integ.tableau.order

# FSAL carry-over: reuse the last stage's derivative as next step's first
# stage, when the tableau supports it. No-op for non-FSAL methods.
fsal_carryover!(integ::FIIntegrator) =
    (integ.tableau.fsal && copyto!(integ.ks[1], integ.ks[end]); nothing)

# `steplog` entry for one accepted step. Tableau methods log `(t, dt)`;
# `FIRKCIntegrator` widens this to `(t, dt, s)` (roadmap §5) without changing
# this method.
steplog_entry(integ::FIIntegrator, dt) = (integ.t, dt)

# Optional periodic spectral-radius re-estimation hook (roadmap §5); a no-op
# for tableau methods, which have no spectral-radius state.
maybe_reestimate!(::FIIntegrator) = nothing

# -----------------------------------------------------------------------------
# Step-size controller (Hairer's PI controller, cf. dopri5)
# -----------------------------------------------------------------------------

const _SAFE = 0.9
const _BETA = 0.04
const _FACMIN = 0.2      # smallest allowed dtnew/dt
const _FACMAX = 10.0     # largest allowed dtnew/dt

function controller_accept_dt(integ::AbstractFIIntegrator, err, dt)
    order = stepper_order(integ)
    expo1 = 1 / order - _BETA * 0.75
    err = max(err, 1e-10)
    fac11 = err^expo1
    fac = fac11 / integ.facold^_BETA
    fac = max(1 / _FACMAX, min(1 / _FACMIN, fac / _SAFE))
    return dt / fac
end

function controller_reject_dt(integ::AbstractFIIntegrator, err, dt)
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
function solve_to!(integ::AbstractFIIntegrator, target, maxiters, steplog = nothing)
    isadaptive(integ.alg) ? solve_to_adaptive!(integ, target, maxiters, steplog) :
        solve_to_fixed!(integ, target, maxiters, steplog)
end

function solve_to_adaptive!(integ::AbstractFIIntegrator, target, maxiters, steplog = nothing)
    T = typeof(integ.t)
    target = T(target)
    iters = 0
    while integ.t < target
        iters += 1
        iters > maxiters && error("FIIntegrator: exceeded maxiters=$maxiters at t=$(integ.t)")

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
                integ.dt = clamp(controller_accept_dt(integ, err, dt), integ.dtmin, integ.dtmax)
                integ.facold = max(err, 1e-4)
            end
            maybe_reestimate!(integ)
        else
            # reject: shrink and retry (u and t unchanged, so ks[1] stays valid)
            integ.nreject += 1
            integ.dt = max(controller_reject_dt(integ, err, dt), integ.dtmin)
        end
    end
    return integ
end

function solve_to_fixed!(integ::FIIntegrator, target, maxiters, steplog = nothing)
    T = typeof(integ.t)
    target = T(target)
    iters = 0
    while integ.t < target
        iters += 1
        iters > maxiters && error("FIIntegrator: exceeded maxiters=$maxiters at t=$(integ.t)")

        dt = min(integ.dt, target - integ.t)
        perform_step!(integ, dt)
        steplog === nothing || push!(steplog, (integ.t, dt))
        integ.t += dt
        copyto!(integ.u, integ.unew)
        # Euler is not FSAL: refresh the first stage for the next step.
        integ.f!(integ.ks[1], integ.u, integ.p, integ.t)
        integ.nf += 1
    end
    return integ
end

# =============================================================================
# FIRKC: stabilised explicit Runge-Kutta-Chebyshev (RKC2, damped, s stages)
#
# Not a Butcher tableau: the stage count `s` is chosen per step from `dt` and
# an estimated spectral radius, and the update is a three-term Chebyshev
# recurrence rather than a fixed stage matrix. See `FIRKC`'s docstring and
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
    @inbounds for j in 2:s
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
    @inbounds for j in 2:s
        Fjm1 = z * y1
        ynext = mu[j] * y1 + nu[j] * y2 + (1 - mu[j] - nu[j]) * y0 +
            mutilde[j] * Fjm1 + gammatilde[j] * F0
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
    stable(beta) = abs(rkc_scalar_map(mu, nu, mutilde, gammatilde, s, -beta)) <= 1 + sqrt(eps(T))
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
    s >= 2 || throw(ArgumentError("FIRKC requires at least 2 stages (got s=$s)"))
    damping > 0 || throw(ArgumentError(
        "FIRKC requires damping > 0 (w1's closed form is singular at the undamped limit w0=1)"))
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
    @inbounds for j in 2:s
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
function rkc_choose_stages(dt, lambda_max::T, damping::T, smax::Int; safety::T = T(1.2)) where {T}
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
# FIRKC integrator state
# -----------------------------------------------------------------------------

"""
    FIRKCIntegrator

Mutable state and O(1) (independent of stage count) work arrays for `FIRKC`.
The rolling Chebyshev recurrence needs only three grid-sized buffers for the
`Y_{j-2}, Y_{j-1}, Y_j` sequence (`ym2`, `ym1`, `unew`, cycled by reference
swap — no per-step allocation) plus `F0` (the RHS at `Y_0`, constant through a
step) and `Fj` (the RHS at the current stage). The per-step coefficient
vectors from `rkc_coeffs` are small (`O(s)` scalars, `s <= smax`, a few KB at
most) and are *not* part of this O(1)-work-array guarantee, which concerns the
grid-sized state only.
"""
mutable struct FIRKCIntegrator{A, T, F, P} <: AbstractFIIntegrator
    f!::F
    p::P
    alg::FIRKC
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

function init_fi(f!, u0::A, tspan, alg::FIRKC, p = nothing;
    reltol = 1e-5,
    abstol = 1e-6,
    dt0 = nothing,
    dtmin = nothing,
    dtmax = nothing,
    lambda_maxiter::Int = 100,
    lambda_tol = 1e-2,
) where {A}
    T = eltype(u0)

    t0, tend = T(tspan[1]), T(tspan[2])
    span = tend - t0

    dt = dt0 === nothing ? abs(span) / 10_000 : T(dt0)
    dtmx = dtmax === nothing ? abs(span) : T(dtmax)
    dtmn = dtmin === nothing ? eps(T) * max(abs(t0), abs(tend)) : T(dtmin)
    dt = clamp(dt, dtmn, dtmx)

    counted_f!, nf0 = _counting_wrapper(f!)
    lambda_max = spectral_radius_estimate(counted_f!, u0, p, t0;
        maxiter = lambda_maxiter, tol = T(lambda_tol))

    return FIRKCIntegrator(
        f!, p, alg,
        t0, dt,
        copy(u0), similar(u0), similar(u0), similar(u0), similar(u0), similar(u0), similar(u0),
        T(lambda_max), 0,
        T(reltol), T(abstol), dtmn, dtmx,
        T(1e-4), 0, 0, nf0[],
    )
end

# -----------------------------------------------------------------------------
# FIRKC single step
# -----------------------------------------------------------------------------

# Same contract as the tableau `perform_step!`: `u`/`t` are left untouched, the
# candidate solution is left in `integ.unew`, and the scaled RMS error norm is
# returned; the driver decides whether to accept.
function perform_step!(integ::FIRKCIntegrator, dt)
    alg = integ.alg
    p, t = integ.p, integ.t
    u = integ.u
    T = eltype(u)

    s = rkc_choose_stages(T(dt), integ.lambda_max, T(alg.damping), alg.smax;
        safety = T(alg.safety))
    integ.s = s
    mu, nu, mutilde, gammatilde, c = rkc_coeffs(T, s, T(alg.damping))

    F0, Fj = integ.F0, integ.Fj
    integ.f!(F0, u, p, t)
    integ.nf += 1

    y0 = u                                             # Y_0, never mutated below
    y1, y2, ynext = integ.ym1, integ.ym2, integ.unew    # rolling Y_{j-1}, Y_{j-2}, scratch
    @. y1 = y0 + mutilde[1] * dt * F0                   # Y_1
    copyto!(y2, y0)                                     # Y_0 snapshot, rolled as "Y_{j-2}" from j=2

    @inbounds for j in 2:s
        integ.f!(Fj, y1, p, t + c[j-1] * dt)
        integ.nf += 1
        muj, nuj, mtj, gtj = mu[j], nu[j], mutilde[j], gammatilde[j]
        @. ynext = muj * y1 + nuj * y2 + (1 - muj - nuj) * y0 + mtj * dt * Fj + gtj * dt * F0
        y2, y1, ynext = y1, ynext, y2
    end
    # Y_s now lives in `y1` (the final rotation moved the last-computed stage there).

    # Embedded low-order (Euler) reference error indicator: zero extra RHS
    # evaluations (F0 already computed above), reuses the same scaled RMS norm
    # as the tableau path. This is a deliberate simplification of "the SSV
    # paper's embedded estimate" (roadmap §5) — a legitimate O(dt) truncation
    # proxy either way (same embedded-pair paradigm as `btilde` for BS3/Tsit5:
    # the *lower*-order method's error controls the step while the higher-order
    # solution is propagated), not guaranteed bit-identical to SSV's own
    # internal calibration. Safety is unaffected regardless: stability comes
    # from `rkc_choose_stages`, not from this estimate. It IS measurably more
    # conservative than FITsit5's embedded estimate at a given `reltol`
    # (roadmap §6 benchmark: ~2-3 orders of magnitude looser in practice) — a
    # real, deliberate scope simplification, not a correctness bug (the
    # recurrence itself is independently verified 2nd order in
    # `test/test_integrators.jl`). Pick a visibly tighter `reltol` than you
    # would for `FITsit5` if comparable global accuracy is required.
    atmp = integ.atmp
    @. atmp = (y1 - (y0 + dt * F0)) / (integ.abstol + integ.reltol * max(abs(y0), abs(y1)))
    err = norm(atmp) / sqrt(length(atmp))

    integ.unew, integ.ym1, integ.ym2 = y1, y2, ynext
    return err
end

# -----------------------------------------------------------------------------
# FIRKC generic-hook methods
# -----------------------------------------------------------------------------

stepper_order(::FIRKCIntegrator) = 2                # RKC2 is second order

fsal_carryover!(::FIRKCIntegrator) = nothing         # not FSAL (roadmap §2.3)

# Widen the steplog tuple to include the stage count used, for a future frozen
# replay (roadmap §5/§7); FIEuler/FIBS3/FITsit5 logging is untouched (their
# `steplog_entry` method above still returns `(t, dt)`).
steplog_entry(integ::FIRKCIntegrator, dt) = (integ.t, dt, integ.s)

function maybe_reestimate!(integ::FIRKCIntegrator)
    n = integ.alg.reestimate_every
    (n > 0 && integ.naccept % n == 0) || return nothing
    counted_f!, nf0 = _counting_wrapper(integ.f!)
    integ.lambda_max = spectral_radius_estimate(counted_f!, integ.u, integ.p, integ.t)
    integ.nf += nf0[]
    return nothing
end

# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

"""
    fi_solve(f!, u0, tspan, alg::FIAlgorithm, p = nothing; kwargs...) -> (ts, us)

Integrate the in-place ODE `f!(du, u, p, t)` from `tspan[1]` to `tspan[2]`.

Returns the sorted save times `ts` and a vector `us` of solution snapshots
(copies) at those times.

Keyword arguments:
- `reltol`, `abstol`: error tolerances for the adaptive controller.
- `saveat`: times at which to store the solution (defaults to `tspan[2]`).
  Must lie within `tspan`. The endpoint is always included.
- `dt0`: initial step size (also *the* step for `FIEuler`).
- `dtmin`, `dtmax`: bounds on the adaptive step size.
- `maxiters`: safety cap on the number of steps per save interval.
"""
function fi_solve(f!, u0, tspan, alg::FIAlgorithm, p = nothing;
    reltol = 1e-5,
    abstol = 1e-6,
    saveat = nothing,
    dt0 = nothing,
    dtmin = nothing,
    dtmax = nothing,
    maxiters = 10_000_000,
)
    T = eltype(u0)
    integ = init_fi(f!, u0, tspan, alg, p;
        reltol = reltol, abstol = abstol, dt0 = dt0, dtmin = dtmin, dtmax = dtmax)

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
# stepper. They dispatch on the *abstract* `FIAlgorithm`, so `FIBS3`/`FITsit5`/
# `FIEuler` all share one path. Output is written by reusing `nc_affect!` and
# `nout_affect!`, which only read `integrator.p` (== the `Simulation`) and
# `integrator.t` — both of which `FIIntegrator` provides.
# =============================================================================

const STEPPER_MAXITERS = 10_000_000

# Build a primed integrator from a Simulation's options. `init_problem!` and
# `t_computation_0` must already have been set by the caller.
function build_integrator(sim)
    opts = sim.opts.diffeq
    alg = opts.alg
    T = eltype(sim.now.u)

    dtmin = opts.dt_min isa Real ? T(opts.dt_min) : nothing
    if alg isa FIEuler
        opts.dt_min isa Real ||
            error("FIEuler requires `DiffEqOptions(dt_min = ...)` (fixed step size).")
        dt0 = T(opts.dt_min)
    else
        dt0 = opts.dt0 isa Real ? T(opts.dt0) : nothing
    end

    return init_fi(update_diagnostics!, sim.now.u, sim.timer.t_span, alg, sim;
        reltol = opts.reltol, abstol = opts.abstol, dt0 = dt0, dtmin = dtmin)
end

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
    tn = (length(sim.nout.t) >= 1 && sim.nout.k <= length(sim.nout.t)) ?
        sim.nout.t[sim.nout.k] : nothing
    tc = (length(sim.ncout.t) >= 1 && sim.ncout.k <= length(sim.ncout.t)) ?
        sim.ncout.t[sim.ncout.k] : nothing
    ts = _next_simobs_time(sim)
    t = tn === nothing ? tc : (tc === nothing ? tn : min(tn, tc))
    return t === nothing ? ts : (ts === nothing ? t : min(t, ts))
end

# Advance the integrator up to `target`, stopping exactly on every output time
# in between and writing output there.
function advance_with_output!(integ::AbstractFIIntegrator, sim, target, maxiters = STEPPER_MAXITERS)
    T = typeof(integ.t)
    target = T(target)
    while true
        te = _next_output_time(sim)
        if te === nothing || te > target
            solve_to!(integ, target, maxiters)
            return integ
        end
        solve_to!(integ, te, maxiters)
        # Fire netCDF first then native output (matches previous callback order),
        # then any simulated observables pending at this time.
        if length(sim.ncout.t) >= 1 && sim.ncout.k <= length(sim.ncout.t) &&
                sim.ncout.t[sim.ncout.k] == te
            nc_affect!(integ)
        end
        if length(sim.nout.t) >= 1 && sim.nout.k <= length(sim.nout.t) &&
                sim.nout.t[sim.nout.k] == te
            nout_affect!(integ)
        end
        for so in sim.simobs
            next_simobs_time(so) == te && record!(so, sim)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Solve the isostatic adjustment problem defined in `sim::Simulation`, integrating
it forward over `sim.timer.t_span` with the algorithm in
`sim.opts.diffeq.alg::FIAlgorithm` and writing output at the requested times.
"""
function run!(sim::Simulation)
    init_problem!(sim)
    sim.timer.t_computation_0 = time()
    integ = build_integrator(sim)
    advance_with_output!(integ, sim, sim.timer.t_span[2])
    isempty(sim.timer.t_computation) ||
        (sim.timer.t_computation .-= sim.timer.t_computation[1])
    return nothing
end

"""
$(TYPEDSIGNATURES)

Initialise the integrator of `sim::Simulation`, which can subsequently be
advanced manually with `step!(integrator, Δt, force_dt)` (e.g. when coupling to
an external ice-sheet model).
"""
function init_integrator(sim::Simulation)
    init_problem!(sim)
    sim.timer.t_computation_0 = time()
    return build_integrator(sim)
end

"""
$(TYPEDSIGNATURES)

Advance `integrator` by the interval `Δt`, using internal adaptive substeps and
writing any output that falls within the interval. `force_dt` is accepted for
backward compatibility; the step always stops exactly at `t + Δt`.
"""
function step!(integ::AbstractFIIntegrator, Δt, force_dt::Bool = true)
    advance_with_output!(integ, integ.p, integ.t + Δt)
    return nothing
end
