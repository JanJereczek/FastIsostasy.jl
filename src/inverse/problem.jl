# =============================================================================
# Inversion problems.
#
# `IceLoadInversion` (application 1: joint ice-load + viscosity over a glacial
# cycle) and `ParameterInversion` (application 2: viscosity + densities calibrated
# against a 3-D GIA model) share the same machinery and differ only in intent /
# defaults. Both hold a `Simulation` template, an encoding, a vector of
# `Observation`s, a collection of regularizations, a diff mode and a loss model.
#
# `loss(prob, θ)` is the plain (AD-free) objective: decode θ into the sim, run the
# forward model stopping at every observation time, accumulate the misfit (via
# `prob.lossmodel`, default `DefaultLoss` = weighted L2) and the regularization
# penalties. The AD engines (extensions) differentiate exactly this function;
# `gradient!` / `solve!` are stubs the extensions implement.
# =============================================================================

abstract type AbstractInversion end

# --- loss model ----------------------------------------------------------------
#
# `AbstractLoss` customizes only the *misfit* term — `misfit(l, preds,
# observations)` — given the predictions already gathered by `forward_predict!`.
# It is stored as a problem field (like `diffmode`), not passed as an extra
# argument to `loss`: `gradient!`/`loss_and_gradient!` shadow the whole `prob` via
# `Enzyme.make_zero`, so a `prob.lossmodel` field (and any arrays it carries, e.g.
# a noise-covariance factor) is picked up automatically with no extension change.
# reconstruct!/forward/regularization orchestration is NOT customizable here —
# only the norm applied to (prediction, data, σ) triples — to keep the
# Enzyme-legality invariants (roadmap §4) in one place (`loss`, below).
abstract type AbstractLoss end

"""
    DefaultLoss()

The weighted-L2 misfit ½·Σₒ‖(predₒ − dataₒ)/σₒ‖², summed over observations.
"""
struct DefaultLoss <: AbstractLoss end

function misfit(::DefaultLoss, preds, observations)
    l = zero(eltype(first(preds)))
    for (k, obs) in enumerate(observations)
        r = (preds[k] .- obs.data) ./ obs.σ
        l += sum(abs2, r) / 2
    end
    return l
end

struct IceLoadInversion{S, E, OB, RG, DM, T, LI, LM} <: AbstractInversion
    sim::S
    encoding::E
    observations::OB
    regularizations::RG
    diffmode::DM
    extract_times::Vector{T}
    extract_plan::Vector{Vector{Tuple{Int, Int}}}
    linear_indices::LI
    lossmodel::LM
end

struct ParameterInversion{S, E, OB, RG, DM, T, LI, LM} <: AbstractInversion
    sim::S
    encoding::E
    observations::OB
    regularizations::RG
    diffmode::DM
    extract_times::Vector{T}
    extract_plan::Vector{Vector{Tuple{Int, Int}}}
    linear_indices::LI
    lossmodel::LM
end

# Sorted unique union of all observation times.
function _extract_times(observations)
    T = eltype(first(observations).times)
    ts = T[]
    for obs in observations
        append!(ts, obs.times)
    end
    return sort!(unique!(ts))
end

# For each extract time, the list of `(obs_index, time_index)` pairs that fall on it.
# Precomputed (θ-independent) so the differentiated `forward_predict!` needs no
# `findfirst` (which returns a `Union{Nothing,Int}` and trips Enzyme).
function _extract_plan(observations, extract_times)
    plan = [Tuple{Int, Int}[] for _ in extract_times]
    for (i, t) in enumerate(extract_times)
        for (k, obs) in enumerate(observations)
            j = findfirst(==(t), obs.times)
            j === nothing || push!(plan[i], (k, j))
        end
    end
    return plan
end

# Backend-promoted linear indices for one observation: `obs.points`
# (`CartesianIndex{2}`, the user-facing API) converted once to linear indices and
# copied onto the same array family as `field` (`Matrix` on CPU, `CuMatrix` on
# GPU via `similar`), so `gather!`'s GPU branch never receives a host `Vector`
# alongside a device `field`.
_linear_indices(obs::Observation, field) = points_to_linear_indices(obs.points, field)

_obs_linear_indices(observations, sim) =
    [_linear_indices(obs, observable_field(obs.tag, sim)) for obs in observations]

function _check_encoding(encoding, diffmode)
    if requires_encoding(diffmode) && encoding === nothing
        throw(ArgumentError(
            "$(typeof(diffmode)) requires an encoding (forward mode is only " *
            "affordable for low-dimensional θ)."))
    end
    return nothing
end

# `Test1Encoding` writes its K = length(knot_times) Vialov snapshots directly into
# `ice_snapshots(sim)[1:K]` (encodings.jl); silently wrong if the sim's ice
# interpolation doesn't have exactly K snapshots at exactly those times. Other
# encodings don't touch the ice snapshots, so nothing to check.
_check_ice_snapshots(sim, ::Nothing) = nothing
_check_ice_snapshots(sim, ::AbstractEncoding) = nothing

function _check_ice_snapshots(sim, enc::Test1Encoding)
    snaps = ice_snapshots(sim)
    K = length(enc.knot_times)
    length(snaps) == K || throw(ArgumentError(
        "Test1Encoding has $K knot_times but the sim's ice-thickness interpolation " *
        "holds $(length(snaps)) snapshots — reconstruct! would write the wrong " *
        "number of them."))
    itp_t = sim.bcs.ice_thickness.H_itp.t
    itp_t == enc.knot_times || throw(ArgumentError(
        "Test1Encoding.knot_times $(enc.knot_times) does not match the sim's ice " *
        "interpolation times $(itp_t) — reconstruct! would silently write ice " *
        "thicknesses at the wrong times."))
    return nothing
end

# Every observation time must fall inside the sim's t_span: before t_span[1] is a
# silent wrong-time extraction (extract_plan only matches times that occur during
# the forward run), after t_span[2] silently extends the run past its intended end.
function _check_obs_times(sim, observations)
    t0, t1 = sim.timer.t_span
    for obs in observations
        all(t -> t0 <= t <= t1, obs.times) || throw(ArgumentError(
            "Observation times must lie within the sim's t_span = ($t0, $t1); " *
            "got times outside that range in $(typeof(obs.tag))."))
    end
    return nothing
end

for Inv in (:IceLoadInversion, :ParameterInversion)
    @eval function $Inv(sim, encoding, observations;
            regularizations = (), diffmode = TangentMode(), lossmodel = DefaultLoss())
        _check_encoding(encoding, diffmode)
        _check_ice_snapshots(sim, encoding)
        obs = collect(observations)
        _check_obs_times(sim, obs)
        times = _extract_times(obs)
        return $Inv(sim, encoding, obs, regularizations, diffmode,
            times, _extract_plan(obs, times), _obs_linear_indices(obs, sim), lossmodel)
    end
end

# --- forward prediction ------------------------------------------------------

# Allocate one prediction vector per observation, on the same array family as the
# sim's fields (so `gather!`'s GPU branch writes into a device array, not a host one).
function allocate_predictions(prob::AbstractInversion)
    T = eltype(prob.extract_times)
    ref = prob.sim.now.u
    return [similar(ref, T, nentries(obs)) for obs in prob.observations]
end

# Fixed-step explicit-Euler advance from `t` to `target`, mathematically identical
# to the `FIEuler` integrator path (`uₖ₊₁ = uₖ + h·f(uₖ,tₖ)`), but Enzyme-legal: it
# avoids the `FIIntegrator`'s `Vector{Matrix}` stage buffers and deep nested type,
# which exceed Enzyme's static type analysis. `update_diagnostics!` (= the RHS) syncs
# `sim.now` each eval (`update_bedrock!`: `sim.now.u .= u`); a final eval at `target`
# leaves `sim.now` at the save state so `extract!` reads current diagnostics.
function _advance_euler!(sim, u, dudt, t, target, dt)
    while t < target
        h = min(dt, target - t)
        update_diagnostics!(dudt, u, sim, t)
        @. u = u + h * dudt
        t += h
    end
    update_diagnostics!(dudt, u, sim, t)
    return t
end

# Run the forward model, stopping at each observation time, and fill `preds` (one
# vector per observation). Dispatches on the algorithm: fixed-step `FIEuler` uses the
# AD-legal direct loop; adaptive algorithms use the stateful integrator (not
# differentiable — TangentMode v1 is fixed-step only). Static dispatch keeps Enzyme
# from analysing the integrator branch when the sim is `FIEuler`.
function forward_predict!(preds, prob::AbstractInversion)
    return _forward_run!(preds, prob, prob.sim.opts.diffeq.alg)
end

function _forward_run!(preds, prob::AbstractInversion, ::FIEuler)
    sim = prob.sim
    dt = sim.opts.diffeq.dt_min
    reset_state!(sim)          # restart from the initial condition each evaluation
    init_problem!(sim)
    u = copy(sim.now.u)
    dudt = similar(u)
    t = sim.timer.t_span[1]
    for (i, tsave) in enumerate(prob.extract_times)
        t = _advance_euler!(sim, u, dudt, t, tsave, dt)
        for (k, j) in prob.extract_plan[i]
            extract!(preds[k], prob.observations[k], j, sim, prob.linear_indices[k])
        end
    end
    return preds
end

function _forward_run!(preds, prob::AbstractInversion, alg)
    sim = prob.sim
    reset_state!(sim)
    init_problem!(sim)
    integ = build_integrator(sim)
    for (i, t) in enumerate(prob.extract_times)
        solve_to!(integ, t, STEPPER_MAXITERS)
        for (k, j) in prob.extract_plan[i]
            extract!(preds[k], prob.observations[k], j, sim, prob.linear_indices[k])
        end
    end
    return preds
end

# --- objective ---------------------------------------------------------------

"""
    loss(prob, θ) -> scalar

Decode `θ` into `prob.sim`, run the forward model to every observation time,
and return `misfit(prob.lossmodel, preds, prob.observations)` plus the
regularization penalties. This is the AD-free objective the diff engines
differentiate.
"""
function loss(prob::AbstractInversion, θ)
    if prob.encoding !== nothing
        length(θ) == nparams(prob.encoding) || throw(DimensionMismatch(
            "θ has length $(length(θ)) but $(typeof(prob.encoding)) expects " *
            "nparams = $(nparams(prob.encoding)) (a too-long θ would otherwise " *
            "be silently truncated by reconstruct!'s indexed reads)."))
    end
    reconstruct!(prob.sim, θ, prob.encoding)
    preds = allocate_predictions(prob)
    forward_predict!(preds, prob)
    l = misfit(prob.lossmodel, preds, prob.observations)
    for reg in prob.regularizations
        l += penalty(reg, prob.sim, θ)
    end
    return l
end

# --- stubs implemented by extensions -----------------------------------------

"""
    gradient!(g, prob, θ)

In-place gradient of `loss(prob, θ)` w.r.t. `θ`. Dispatches on `prob.diffmode`
(`TangentMode` → forward, `AdjointMode` → checkpointed reverse), implemented by
the Enzyme extension. The 3-arg dispatcher lives in core (so it always resolves);
the mode-specific 4-arg method falls back to an informative error here and is
overridden by the extension once loaded.
"""
function gradient!(g, prob::AbstractInversion, θ)
    return gradient!(g, prob, θ, prob.diffmode)
end

function gradient!(g, prob::AbstractInversion, θ, mode::AbstractDiffMode)
    error("gradient!(::$(typeof(mode))) requires FastIsostasyEnzymeExt to be " *
          "loaded (`using Enzyme`) — the core package doesn't implement AD itself.")
end

"""
    loss_and_gradient!(g, prob, θ) -> loss_value

In-place gradient of `loss(prob, θ)` w.r.t. `θ`, written into `g`, returning the
loss value alongside it. Under `TangentMode` the primal falls out of the same
forward pass that computes each directional tangent (`Enzyme.ForwardWithPrimal`),
so this is one fewer forward run per optimizer iteration than calling `loss` and
`gradient!` separately. Implemented by the Enzyme extension; see `gradient!` for
the dispatcher/fallback split.
"""
function loss_and_gradient!(g, prob::AbstractInversion, θ)
    return loss_and_gradient!(g, prob, θ, prob.diffmode)
end

function loss_and_gradient!(g, prob::AbstractInversion, θ, mode::AbstractDiffMode)
    error("loss_and_gradient!(::$(typeof(mode))) requires FastIsostasyEnzymeExt " *
          "to be loaded (`using Enzyme`) — the core package doesn't implement AD " *
          "itself.")
end

"""
    solve!(prob, optimizer; kwargs...)

Run the inversion with `optimizer`. Implemented by the Optim extension.
"""
function solve! end
