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
mutable struct FIIntegrator{A, T, F, P, Alg <: FIAlgorithm}
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
# Step-size controller (Hairer's PI controller, cf. dopri5)
# -----------------------------------------------------------------------------

const _SAFE = 0.9
const _BETA = 0.04
const _FACMIN = 0.2      # smallest allowed dtnew/dt
const _FACMAX = 10.0     # largest allowed dtnew/dt

function controller_accept_dt(integ::FIIntegrator, err, dt)
    order = integ.tableau.order
    expo1 = 1 / order - _BETA * 0.75
    err = max(err, 1e-10)
    fac11 = err^expo1
    fac = fac11 / integ.facold^_BETA
    fac = max(1 / _FACMAX, min(1 / _FACMIN, fac / _SAFE))
    return dt / fac
end

function controller_reject_dt(integ::FIIntegrator, err, dt)
    order = integ.tableau.order
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
function solve_to!(integ::FIIntegrator, target, maxiters, steplog = nothing)
    isadaptive(integ.alg) ? solve_to_adaptive!(integ, target, maxiters, steplog) :
        solve_to_fixed!(integ, target, maxiters, steplog)
end

function solve_to_adaptive!(integ::FIIntegrator, target, maxiters, steplog = nothing)
    tab = integ.tableau
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
            steplog === nothing || push!(steplog, (integ.t, dt))
            integ.t += dt
            copyto!(integ.u, integ.unew)
            tab.fsal && copyto!(integ.ks[1], integ.ks[end])   # FSAL carry-over
            integ.naccept += 1
            # Only let a *natural* (unclipped) step drive the controller, so
            # clipping onto a save time does not shrink the running dt.
            if !clipped
                integ.dt = clamp(controller_accept_dt(integ, err, dt), integ.dtmin, integ.dtmax)
                integ.facold = max(err, 1e-4)
            end
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
function advance_with_output!(integ::FIIntegrator, sim, target, maxiters = STEPPER_MAXITERS)
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
function step!(integ::FIIntegrator, Δt, force_dt::Bool = true)
    advance_with_output!(integ, integ.p, integ.t + Δt)
    return nothing
end
