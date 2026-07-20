# =============================================================================
# Forward recording for the discrete adjoint (roadmap §6, Phase 5).
#
# The reverse sweep needs two things from the (non-differentiated) forward pass:
#   1. a `StateSnapshot` at each save/observation-interval boundary, so any
#      interval can be recomputed from its start without rerunning from t0;
#   2. the accepted `(tₖ, dtₖ)` sequence inside each interval, recorded exactly
#      as evaluated and **frozen** during the reverse pass (the PI controller is
#      never differentiated).
#
# `record_forward!` is `forward_predict!` plus this bookkeeping: it runs the same
# operation sequence (same RHS evaluations at the same times), so its predictions
# are bit-identical to `forward_predict!`'s. `replay_interval!` restores a
# checkpoint and re-runs one interval with the frozen dt-sequence — the primitive
# the checkpointing extension will differentiate step by step. Intervals are the
# gaps between consecutive `prob.extract_times` (the first starts at t_span[1]);
# nothing after the last observation time affects the loss, so nothing after it
# is recorded.
#
# Replay is currently `FIEuler`-only, mirroring `_forward_run!`'s direct-Euler
# loop. Adaptive algorithms record fine (via the integrator's `steplog`), but
# replaying them needs the integrator's FSAL state — `integ.u` differs bitwise
# from the projected `sim.now.u` after an accepted step — which is deferred to
# the adaptive-adjoint work (roadmap Phase 5+).
# =============================================================================

"""
    ForwardRecord(prob::AbstractInversion)

Preallocated recording buffers for `record_forward!`: one `StateSnapshot` per
save/observation-interval boundary (`length(prob.extract_times) + 1`, including
the state at the end of the run) and one accepted-steplog-entry vector per
interval. The entry type `E` follows `prob.sim.opts.diffeq.alg` via
`steplog_entry_type` (`Tuple{T,T}` for the tableau algorithms, widened to
`Tuple{T,T,Int}` for `FIRKC` — see `steplog_entry`/`steplog_entry_type` in
`src/integrators.jl`), so `push!`ing whatever `steplog_entry` a given
algorithm's integrator produces always matches the buffer's element type.
Reusable across recordings — step vectors are emptied, snapshots overwritten.
"""
struct ForwardRecord{T, SS, E}
    checkpoints::Vector{SS}      # [i] = state at the start of interval i;
                                 # [end] = state at the end of the run
    steps::Vector{Vector{E}}     # accepted steplog entries per interval
end

function ForwardRecord(prob::AbstractInversion)
    T = promote_type(eltype(prob.sim.now.u), eltype(prob.extract_times))
    n = length(prob.extract_times)
    checkpoints = [StateSnapshot(prob.sim) for _ in 1:(n + 1)]
    E = steplog_entry_type(prob.sim.opts.diffeq.alg, T)
    steps = [E[] for _ in 1:n]
    return ForwardRecord{T, eltype(checkpoints), E}(checkpoints, steps)
end

nintervals(rec::ForwardRecord) = length(rec.steps)

"""
    record_forward!(preds, rec::ForwardRecord, prob::AbstractInversion) -> preds

Run the forward model exactly as `forward_predict!` (same RHS evaluations, same
predictions, bit for bit) while snapshotting the state at every interval
boundary into `rec.checkpoints` and logging the accepted `(t, dt)` steps of each
interval into `rec.steps`.
"""
function record_forward!(preds, rec::ForwardRecord, prob::AbstractInversion)
    nintervals(rec) == length(prob.extract_times) || throw(DimensionMismatch(
        "ForwardRecord has $(nintervals(rec)) intervals but the problem has " *
        "$(length(prob.extract_times)) extract times — build the record from " *
        "this problem (`ForwardRecord(prob)`)."))
    return _record_run!(preds, rec, prob, prob.sim.opts.diffeq.alg)
end

# Fixed-step explicit Euler: the recorded twin of `_forward_run!(_, _, ::FIEuler)`
# (problem.jl). The loop scalars are held in the record's step type `T` so the
# logged `(t, h)` are exactly the values passed to the RHS — a frozen replay then
# reproduces every evaluation bitwise.
function _record_run!(preds, rec::ForwardRecord{T}, prob::AbstractInversion,
        ::FIEuler) where {T}
    sim = prob.sim
    dt = T(sim.opts.diffeq.dt_min)
    reset_state!(sim)
    init_problem!(sim)
    u = copy(sim.now.u)
    dudt = similar(u)
    t = T(sim.timer.t_span[1])
    for (i, tsave) in enumerate(prob.extract_times)
        snapshot!(rec.checkpoints[i], sim)
        steps = empty!(rec.steps[i])
        target = T(tsave)
        while t < target
            h = min(dt, target - t)
            update_diagnostics!(dudt, u, sim, t)
            @. u = u + h * dudt
            push!(steps, (t, h))
            t += h
        end
        update_diagnostics!(dudt, u, sim, t)
        for (k, j) in prob.extract_plan[i]
            extract!(preds[k], prob.observations[k], j, sim, prob.linear_indices[k])
        end
    end
    snapshot!(rec.checkpoints[end], sim)
    return preds
end

# Adaptive algorithms: the recorded twin of `_forward_run!(_, _, alg)` — the
# integrator does the stepping and logs its accepted steps via `steplog`.
function _record_run!(preds, rec::ForwardRecord, prob::AbstractInversion, alg)
    sim = prob.sim
    reset_state!(sim)
    init_problem!(sim)
    integ = build_integrator(sim)
    for (i, tsave) in enumerate(prob.extract_times)
        snapshot!(rec.checkpoints[i], sim)
        solve_to!(integ, tsave, STEPPER_MAXITERS, empty!(rec.steps[i]))
        for (k, j) in prob.extract_plan[i]
            extract!(preds[k], prob.observations[k], j, sim, prob.linear_indices[k])
        end
    end
    snapshot!(rec.checkpoints[end], sim)
    return preds
end

"""
    replay_interval!(u, dudt, rec::ForwardRecord, prob::AbstractInversion, i) -> u

Restore `prob.sim` to the checkpoint at the start of interval `i` and re-run
that interval with the frozen `(t, dt)` sequence from `rec.steps[i]`, using `u`
and `dudt` as work buffers (both `similar(sim.now.u)`-shaped). Reproduces the
recorded trajectory bit for bit: afterwards the state matches
`rec.checkpoints[i + 1]` exactly. This is the recomputation primitive the
checkpointing extension reverses step by step.
"""
function replay_interval!(u, dudt, rec::ForwardRecord, prob::AbstractInversion, i)
    return _replay_interval!(u, dudt, rec, prob, i, prob.sim.opts.diffeq.alg)
end

function _replay_interval!(u, dudt, rec::ForwardRecord{T}, prob::AbstractInversion,
        i, ::FIEuler) where {T}
    sim = prob.sim
    restore!(sim, rec.checkpoints[i])
    copyto!(u, sim.now.u)
    for (t, h) in rec.steps[i]
        update_diagnostics!(dudt, u, sim, t)
        @. u = u + h * dudt
    end
    update_diagnostics!(dudt, u, sim, T(prob.extract_times[i]))
    return u
end

function _replay_interval!(u, dudt, rec::ForwardRecord, prob::AbstractInversion, i, alg)
    error("replay_interval! currently supports FIEuler only: replaying an " *
          "adaptive $(typeof(alg)) interval needs the integrator's FSAL state " *
          "(`integ.u` differs bitwise from the projected `sim.now.u`), which is " *
          "deferred to the adaptive-adjoint work (roadmap Phase 5+).")
end
