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
# FIEuler only (like TangentMode): the adaptive integrator's `Vector{Matrix}`
# stage buffers overflow Enzyme's static type analysis, and adaptive replay also
# needs the integrator's FSAL state (roadmap §6). Guarded up front.

using Enzyme: Enzyme, Const, Duplicated, Active
# Checkpointing is a declared trigger of this extension (periodic-schedule
# checkpointing is the design target); the current in-memory `ForwardRecord`
# realises the periodic schedule directly (one checkpoint per save interval), so
# the package is loaded for the trigger/version bound but not yet called into.
# Swapping the dense in-memory record for a Checkpointing.Periodic/Revolve
# schedule over many intervals is the on-disk / large-nstep follow-up.
using Checkpointing: Checkpointing

import FastIsostasy: gradient!, loss_and_gradient!, AbstractInversion, AdjointMode
import FastIsostasy

# --- per-interval primal (reversed by Enzyme) --------------------------------

# Pure in-place replay of interval `i`'s frozen `(t, dt)` steps, ending with a
# diagnostics eval at `tend` (mirrors `record_forward!`/`replay_interval!` so the
# primal is bit-identical). Local `u`/`dudt`; mutates `sim.now.*` in place.
# `steps` and `tend` are sim-free `Const`s — passing `prob` here would alias the
# `Duplicated` sim.
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

# --- observation cotangent seeding -------------------------------------------

# Add the adjoint of one observed field entry (`g = ∂misfit/∂field[idx]`) to the
# sim shadow, resolving `observable_field`'s composition back to the *stored*
# `sim.now` arrays it is built from. Linear `idx` matches `gather!`'s indexing.
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

_require_fieuler_adj(prob) = prob.sim.opts.diffeq.alg isa FastIsostasy.FIEuler || error(
    "AdjointMode is fixed-step only: gradient! requires " *
    "prob.sim.opts.diffeq.alg isa FIEuler (got $(typeof(prob.sim.opts.diffeq.alg))). " *
    "Adaptive-step adjoints (frozen dt + integrator FSAL state) are roadmap Phase 5+.")

# Core: fill `g` with ∇_θ loss(prob, θ) via the checkpointed reverse; return the
# primal loss (misfit + regularization) as a byproduct of the forward record.
function _adjoint_gradient!(g, prob::AbstractInversion, θ)
    _require_fieuler_adj(prob)
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
    dsim = Enzyme.make_zero(sim)
    n = length(prob.extract_times)
    for i in n:-1:1
        _seed_interval!(dsim, prob, preds, i)
        FastIsostasy.restore!(sim, rec.checkpoints[i])
        Enzyme.autodiff(mode, _interval_forward!, Const,
            Duplicated(sim, dsim),
            Const(rec.steps[i]),
            Const(oftype(sim.timer.t, prob.extract_times[i])))
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

end # module
