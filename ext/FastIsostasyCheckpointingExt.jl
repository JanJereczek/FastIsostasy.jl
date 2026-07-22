module FastIsostasyCheckpointingExt

# Checkpointed reverse-mode adjoint of the forward model (roadmap Phase 5).
#
# `gradient!(g, prob, θ, ::AdjointMode)` — one reverse sweep gives the full
# gradient (cost independent of nθ, unlike TangentMode's nθ forward passes), with
# memory bounded to a single save interval via checkpointing.
#
# Discrete-adjoint structure. `θ` enters the model exactly once (via
# `reconstruct!`, which writes `effective_viscosity`/densities/ice); those param
# arrays are then *read at every step* inside `update_diagnostics!` but never
# written. So under reverse mode their cotangents *accumulate* into the sim
# shadow `dsim` across all intervals, while the evolving state's cotangent
# *threads* between interval boundaries. Because `update_bedrock!` syncs
# `sim.now.u .= u` every RHS eval, the full threaded state lives in `sim.now`
# (+ BSL/scalars), all of which `dsim` shadows.
#
# The sweep (i = n → 1):
#   1. seed the *end-of-interval-i* state cotangent in `dsim` from the recorded
#      misfit residual of the observations at that time (`_seed_interval!`);
#   2. `restore!` the sim primal to checkpoint i (interval start);
#   3. `Enzyme.autodiff(Reverse, _interval_forward!, Duplicated(sim, dsim), …)`.
#      `_interval_forward!` is a *pure in-place* replay of interval i's frozen
#      steps; Enzyme's augmented-primal recomputes+tapes the interval (this is the
#      "recompute forward storing every step" the roadmap describes, bounded to
#      one interval), and its reverse transforms `dsim` from cotangent-of-end-state
#      into cotangent-of-start-state — the boundary threading, for free.
# After the sweep, `dsim`'s param fields hold ∂misfit/∂params; one reverse of
# `reconstruct!` maps them to ∂misfit/∂θ. Regularization gradients are added
# per-reg (each its own concrete-typed reverse, so a heterogeneous reg collection
# never trips Enzyme's dynamic-dispatch analysis).
#
# `_interval_forward!` must NOT receive `prob` (which holds `sim`): that would
# alias the `Duplicated` sim argument. It takes only sim-free `Const` pieces
# (the frozen `(t, dt)` vector and the interval-end time).
#
# EulerIntegrator only. The FSAL tableau methods (BS3Integrator/Tsit5Integrator)
# are excluded structurally: adaptive replay needs their integrator FSAL state and
# their `Vector{Matrix}` stage buffers overflow Enzyme's static type analysis.
# RKCIntegrator sidesteps both — it is non-FSAL (a step is a pure function of
# `(u, t, dt, s)`) and its recurrence uses O(1) named stage buffers, replayed via
# `rkc_replay_stages!` — but its reverse transform is currently switched OFF for
# compile-time reasons; see `_reverse_interval!` below. Guarded up front by
# `_require_replayable_adj`.

using Enzyme: Enzyme, Const, Duplicated, Active
# Checkpointing is a declared trigger of this extension (periodic-schedule
# checkpointing is the design target); the current in-memory `ForwardRecord`
# realises the periodic schedule directly (one checkpoint per save interval), so
# the package is loaded for the trigger/version bound but not yet called into.
# Swapping the dense in-memory record for a Checkpointing.Periodic/Revolve
# schedule over many intervals is the on-disk / large-nstep follow-up.
using Checkpointing: Checkpointing
# Extension modules do not inherit the parent's `using`s, so the `$(TYPEDSIGNATURES)`
# interpolations in the docstrings below need their own import. Without it the
# extension fails to *precompile*, and the only symptom is `gradient!` falling back
# to its "reverse mode requires FastIsostasyCheckpointingExt" stub — i.e. a missing
# docstring import masquerades as a missing dependency.
using DocStringExtensions

import FastIsostasy: gradient!, loss_and_gradient!, AbstractInversion, AdjointMode
import FastIsostasy

# --- per-interval primal (reversed by Enzyme) --------------------------------

"""
$(TYPEDSIGNATURES)

Pure in-place replay of an interval's frozen `(t, dt)` steps, ending with a
diagnostics evaluation at `tend`. This mirrors `record_forward!`/`replay_interval!`
so the primal is bit-identical. Uses local `u`/`dudt` and mutates `sim.now.*`
in-place. `steps` and `tend` are expected to be sim-free `Const`s — passing
`prob` here would alias the `Duplicated` sim.
"""
function _interval_forward!(sim, steps, tend)
    u = copy(sim.now.u)
    dudt = similar(u)
    for k in eachindex(steps)
        t, h = steps[k]
        FastIsostasy.update_diagnostics!(dudt, u, sim, t)
        @. u = u + h * dudt
    end
    FastIsostasy.update_diagnostics!(dudt, u, sim, tend)
    return nothing
end

# RKCIntegrator counterpart: replay the frozen `(t, dt, s)` RKC2 recurrence (roadmap §7).
#
# NOT CURRENTLY DIFFERENTIATED — kept, unreferenced by any `Enzyme.autodiff`, so
# the RKC adjoint can be switched back on in one line once its transform is
# affordable (see `_reverse_interval!`). A function Enzyme is never asked to
# transform costs nothing to keep.
#
# Non-FSAL, so — like the Euler replay above — this is a pure in-place function
# of `sim.now` with local scratch; Enzyme reverses it. `rkc_replay_stages!`
# (src/integrators.jl) is the shared recurrence, keeping this bit-identical to
# `record_forward!`/`replay_interval!`. No trailing `tend` sync: each step's own
# error-estimate stage leaves `sim.now` synced at its `t+dt`, and the last step's
# `t+dt` is the interval end (see the recurrence). The O(1) stage buffers are
# six named locals (not a `Vector{Matrix}`), so they stay within Enzyme's static
# type analysis.
#
# `plan` is a FLAT, per-stage `Vector{RKCStage}` built by `rkc_stage_plan`
# *outside* this function and passed as a `Const`. Both properties matter for
# compile time:
#
#   * flat — the differentiated region is a single loop with ONE
#     `update_diagnostics!` call site at depth 1. The earlier
#     loop-over-steps/loop-over-stages form put the RHS at depth 2 under an inner
#     bound (`s`) that varies per step, which forces Enzyme into a jagged
#     two-level tape allocation *per taped value*; see the design note above
#     `RKCStage` in src/integrators.jl.
#   * Const — the coefficient machinery (`rkc_coeffs`: `cosh`/`sinh`/`log`,
#     per-degree Chebyshev tables, allocation) and the stage-time arithmetic
#     `t + c[j-1]*dt` are state-independent, so they carry no cotangent and are
#     hoisted out of the differentiated region entirely.
function _interval_forward_rkc!(sim, plan)
    y1 = copy(sim.now.u)
    y0 = similar(y1); y2 = similar(y1); ynext = similar(y1)
    F = similar(y1); F0 = similar(y1)
    FastIsostasy.rkc_replay_stages!(FastIsostasy.update_diagnostics!,
        y0, y1, y2, ynext, F, F0, sim, plan)
    return nothing
end

# Flat per-stage plan for interval `i`, computed once before the reverse pass
# (state-independent ⇒ `Const` into `Enzyme.autodiff`).
function _rkc_interval_plan(sim, steps)
    T = eltype(sim.now.u)
    return FastIsostasy.rkc_stage_plan(T, steps, T(sim.opts.integ.damping))
end

# --- observation cotangent seeding -------------------------------------------

"""
$(TYPEDSIGNATURES)

Add the adjoint of one observed field entry (`g = ∂misfit/∂field[idx]`) to the
sim shadow, resolving `observable_field`'s composition back to the *stored*
`sim.now` arrays it is built from. Linear `idx` matches `gather!`'s indexing.
"""
@inline function _seed_field!(dsim, ::FastIsostasy.VerticalUpliftObservable, idx, g)
    dsim.now.u[idx] += g            # field = u + ue
    dsim.now.ue[idx] += g
    return nothing
end
@inline function _seed_field!(dsim, ::FastIsostasy.VerticalUpliftRateObservable, idx, g)
    dsim.now.dudt[idx] += g         # field = dudt
    return nothing
end
@inline function _seed_field!(dsim, ::FastIsostasy.RelativeSeaLevelObservable, idx, g)
    dsim.now.z_ss[idx] += g         # field = z_ss − (u + ue)
    dsim.now.u[idx] -= g
    dsim.now.ue[idx] -= g
    return nothing
end

# Seed `dsim` with the end-of-interval-`i` misfit cotangent, from the *recorded*
# model predictions `preds` (so no sim primal is needed here). For a weighted-L2
# misfit ½Σ((pred−data)/σ)², ∂misfit/∂field = (pred−data)/σ².
function _seed_interval!(dsim, prob, preds, i)
    for (k, j) in prob.extract_plan[i]
        obs = prob.observations[k]
        sl = FastIsostasy.time_slice(obs, j)
        lidx = prob.linear_indices[k]
        pk = preds[k]
        for (n, idx) in enumerate(lidx)
            e = sl[n]
            σn = obs.σ isa AbstractVector ? obs.σ[e] : obs.σ
            g = (pk[e] - obs.data[e]) / σn^2
            _seed_field!(dsim, obs.tag, idx, g)
        end
    end
    return nothing
end

# --- the checkpointed reverse sweep ------------------------------------------

# --- per-interval reverse, dispatched on the integrator -----------------------
#
# One method per integrator, DISPATCHED rather than branched. This is load-
# bearing for compile time, not a style choice.
#
# `Enzyme.autodiff` expands a generated function, so Enzyme's LLVM-level reverse
# transform runs for every `autodiff` call site Julia can reach when
# `_adjoint_gradient!` is compiled — including the arms of an `if` that is never
# taken at runtime. Writing this as one function with `if alg isa RKCIntegrator`
# therefore paid for the Euler *and* the RKC transform on every single
# `gradient!(::AdjointMode)`, each a multi-minute LLVM job, no matter which
# integrator the simulation actually used. (It went unnoticed because
# `SolverOptions` used to hold its integrator behind an abstract field, so the
# predicate could not even be const-folded away.) Dispatch leaves Julia exactly
# one method to compile.
#
# Consequence for anyone extending this: a new integrator's reverse is added by
# adding a method here, never by adding a branch to an existing one.

function _reverse_interval!(mode, sim, dsim, ::FastIsostasy.EulerIntegrator, steps, tend)
    Enzyme.autodiff(
        mode, _interval_forward!, Const,
        Duplicated(sim, dsim),
        Const(steps),
        Const(oftype(sim.timer.t, tend)))
    return nothing
end

# RKC: the replay primal (`_interval_forward_rkc!`) is written and bitwise-
# verified, but differentiating it is switched off — its reverse transform costs
# as much as Euler's and doubled every adjoint compile while unvalidated
# (roadmap ad_inversion.md). To re-enable, restore the `Enzyme.autodiff` call on
# `_interval_forward_rkc!` with `Const(_rkc_interval_plan(sim, steps))` here.
_reverse_interval!(_, _, _, ::FastIsostasy.RKCIntegrator, _, _) =
    error(
    "AdjointMode does not currently differentiate RKCIntegrator: its reverse " *
    "transform is deliberately not compiled (it doubled every adjoint compile " *
    "while unvalidated). The frozen replay itself is implemented and verified — " *
    "see `_interval_forward_rkc!` in FastIsostasyCheckpointingExt to switch it " *
    "back on. Use `EulerIntegrator(dt = ...)` for a reverse-mode gradient.")

_reverse_interval!(_, _, _, alg::FastIsostasy.AbstractIntegrator, _, _) =
    error(_unsupported_adj_msg(alg))

# Which integrators have a reverse wired up. Keep in sync with the
# `_reverse_interval!` methods above; used for the up-front guard so an
# unsupported setup fails before paying for a forward record.
_adj_replayable(::FastIsostasy.AbstractIntegrator) = false
_adj_replayable(::FastIsostasy.EulerIntegrator) = true

_unsupported_adj_msg(alg) =
    "AdjointMode replays EulerIntegrator only: gradient! got $(typeof(alg)). " *
    "The FSAL tableau methods (BS3Integrator/Tsit5Integrator) need the " *
    "integrator's FSAL state to replay, and their `Vector{Matrix}` stage " *
    "buffers overflow Enzyme's static type analysis. Use " *
    "`EulerIntegrator(dt = ...)`."

_require_replayable_adj(prob) =
    _adj_replayable(prob.sim.opts.integ) ||
    error(_unsupported_adj_msg(prob.sim.opts.integ))

# Core: fill `g` with ∇_θ loss(prob, θ) via the checkpointed reverse; return the
# primal loss (misfit + regularization) as a byproduct of the forward record.
function _adjoint_gradient!(g, prob::AbstractInversion, θ)
    _require_replayable_adj(prob)
    length(g) == length(θ) || throw(DimensionMismatch(
        "gradient buffer length $(length(g)) ≠ θ length $(length(θ))"))

    # `enc === nothing` is the full-field case (θ is the flattened log10 viscosity
    # field, roadmap Test 3): every step below is encoding-agnostic — the reverse
    # sweep accumulates the `effective_viscosity` cotangent in `dsim`, and the final
    # reverse of `reconstruct!(sim, θ, nothing)` maps it through `10^` to `g`.
    sim = prob.sim
    enc = prob.encoding
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse)

    # Forward record: θ → params (once), then the frozen trajectory + checkpoints
    # + predictions. `record_forward!` calls reset_state!/init_problem! internally
    # but never reconstruct!, so params set here persist through the record.
    FastIsostasy.reconstruct!(sim, θ, enc)
    rec = FastIsostasy.ForwardRecord(prob)
    preds = FastIsostasy.allocate_predictions(prob)
    FastIsostasy.record_forward!(preds, rec, prob)

    # Primal loss (for loss_and_gradient!), read straight off the record.
    lval = FastIsostasy.misfit(prob.lossmodel, preds, prob.observations)
    for reg in prob.regularizations
        lval += FastIsostasy.penalty(reg, sim, θ)
    end

    # Reverse sweep over intervals — accumulates param cotangents in `dsim`.
    # `_reverse_interval!` dispatches to the integrator's own frozen replay;
    # `_require_replayable_adj` above has already rejected the ones without one.
    dsim = Enzyme.make_zero(sim)
    alg = sim.opts.integ
    n = length(prob.extract_times)
    for i in n:-1:1
        _seed_interval!(dsim, prob, preds, i)
        FastIsostasy.restore!(sim, rec.checkpoints[i])
        _reverse_interval!(mode, sim, dsim, alg, rec.steps[i], prob.extract_times[i])
    end

    # Map accumulated ∂misfit/∂params → ∂misfit/∂θ by reversing reconstruct!.
    # `dsim`'s param fields are the output cotangents of reconstruct!; θ's shadow
    # receives the pullback. (sim primal is irrelevant: reconstruct! overwrites
    # params from θ + the Const domain.)
    fill!(g, zero(eltype(g)))
    Enzyme.autodiff(mode, FastIsostasy.reconstruct!, Const,
        Duplicated(sim, dsim), Duplicated(θ, g), Const(enc))

    # Regularization gradients — one concrete-typed reverse per reg (so a mixed
    # reg collection never forces dynamic dispatch through Enzyme). Each reg's
    # penalty may depend on θ directly and/or via sim params (through its own
    # reconstruct!), both captured by `_one_reg_objective`.
    for reg in prob.regularizations
        dθr = zero(θ)
        Enzyme.autodiff(mode, _one_reg_objective, Active,
            Duplicated(sim, Enzyme.make_zero(sim)), Duplicated(θ, dθr),
            Const(reg), Const(enc))
        g .+= dθr
    end

    return lval
end

_one_reg_objective(sim, θ, reg, enc) =
    (FastIsostasy.reconstruct!(sim, θ, enc); FastIsostasy.penalty(reg, sim, θ))

function gradient!(g, prob::AbstractInversion, θ, ::AdjointMode)
    _adjoint_gradient!(g, prob, θ)
    return g
end

function loss_and_gradient!(g, prob::AbstractInversion, θ, ::AdjointMode)
    return _adjoint_gradient!(g, prob, θ)
end

# NOTE — no `@compile_workload` here. Tried (2026-07-11) and measured: a
# `PrecompileTools` workload running `gradient!(::AdjointMode)` during ext
# precompilation does NOT cache the expensive part. The ~665 s cost is Enzyme's
# LLVM-level reverse transformation of `autodiff(Reverse, _interval_forward!, …)`,
# generated through Enzyme's own JIT pipeline, which is not serialized into the
# package image (Enzyme 0.13). Cold-session first `gradient!` stayed ~690 s with
# the workload while precompilation grew by ~739 s — a net loss. (PrecompileTools
# *did* cache the surrounding Julia code — constructors/primal, `t_build` 15→3 s —
# but that is minor.) Enzyme caches within a session, so the practical mitigations
# are a warm dev session, or `TangentMode` (167 s compile) for low-dim encoded θ;
# reverse's 665 s only pays off at full-field scale. Revisit if Enzyme gains
# pkgimage caching of its generated adjoints.
#
# Re-measured 2026-07-21 on the `test_adjoint_validity` setup, cold session:
# `gradient!` 714.5 s cold / ~0 s warm, i.e. the whole cost is the one-off
# transform and none of it is the reverse sweep itself (the primal `loss` compiles
# in 0.1 s). Peak RSS 2.2 GiB. That is back in line with the 665 s above, i.e. the
# single-`autodiff`-call-site cost — see the dispatch note on `_reverse_interval!`
# for why an `alg isa …` branch used to pay it twice per gradient.

end # module
