# =============================================================================
# Inversion problems.
#
# `IceLoadInversion` (application 1: joint ice-load + viscosity over a glacial
# cycle) and `ParameterInversion` (application 2: viscosity + densities calibrated
# against a 3-D GIA model) share the same machinery and differ only in intent /
# defaults. Both hold a `Simulation` template, an encoding, a vector of
# `Observation`s, a collection of regularizations and a diff mode.
#
# `loss(prob, θ)` is the plain (AD-free) objective: decode θ into the sim, run the
# forward model stopping at every observation time, accumulate the data misfit and
# the regularization penalties. The AD engines (extensions) differentiate exactly
# this function; `gradient!` / `solve!` are stubs the extensions implement.
# =============================================================================

abstract type AbstractInversion end

struct IceLoadInversion{S, E, OB, RG, DM, T} <: AbstractInversion
    sim::S
    encoding::E
    observations::OB
    regularizations::RG
    diffmode::DM
    extract_times::Vector{T}
end

struct ParameterInversion{S, E, OB, RG, DM, T} <: AbstractInversion
    sim::S
    encoding::E
    observations::OB
    regularizations::RG
    diffmode::DM
    extract_times::Vector{T}
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

function _check_encoding(encoding, diffmode)
    if requires_encoding(diffmode) && encoding === nothing
        throw(ArgumentError(
            "$(typeof(diffmode)) requires an encoding (forward mode is only " *
            "affordable for low-dimensional θ)."))
    end
    return nothing
end

for Inv in (:IceLoadInversion, :ParameterInversion)
    @eval function $Inv(sim, encoding, observations;
            regularizations = (), diffmode = TangentMode())
        _check_encoding(encoding, diffmode)
        obs = collect(observations)
        return $Inv(sim, encoding, obs, regularizations, diffmode,
            _extract_times(obs))
    end
end

# --- forward prediction ------------------------------------------------------

# Allocate one prediction vector per observation.
function allocate_predictions(prob::AbstractInversion)
    T = eltype(prob.extract_times)
    return [zeros(T, nentries(obs)) for obs in prob.observations]
end

# Run the forward model, stopping at each observation time, and fill `preds`
# (one vector per observation) with the sampled model values.
function forward_predict!(preds, prob::AbstractInversion)
    sim = prob.sim
    reset_state!(sim)          # restart from the initial condition each evaluation
    init_problem!(sim)
    integ = build_integrator(sim)
    for t in prob.extract_times
        advance_with_output!(integ, sim, t)
        for (k, obs) in enumerate(prob.observations)
            j = findfirst(==(t), obs.times)
            j === nothing && continue
            extract!(preds[k], obs, j, sim)
        end
    end
    return preds
end

# --- objective ---------------------------------------------------------------

function data_misfit(prob::AbstractInversion, preds)
    l = zero(eltype(prob.extract_times))
    for (k, obs) in enumerate(prob.observations)
        r = (preds[k] .- obs.data) ./ obs.σ
        l += sum(abs2, r) / 2
    end
    return l
end

"""
    loss(prob, θ) -> scalar

Decode `θ` into `prob.sim`, run the forward model to every observation time,
and return ½·Σ‖(prediction − data)/σ‖² plus the regularization penalties.
This is the AD-free objective the diff engines differentiate.
"""
function loss(prob::AbstractInversion, θ)
    reconstruct!(prob.sim, θ, prob.encoding)
    preds = allocate_predictions(prob)
    forward_predict!(preds, prob)
    l = data_misfit(prob, preds)
    for reg in prob.regularizations
        l += penalty(reg, prob.sim, θ)
    end
    return l
end

# --- stubs implemented by extensions -----------------------------------------

"""
    gradient!(g, prob, θ)

In-place gradient of `loss(prob, θ)` w.r.t. `θ`. Implemented by the Enzyme
extension (dispatch on `prob.diffmode`: `TangentMode` → forward, `AdjointMode`
→ checkpointed reverse). Errors if the extension is not loaded.
"""
function gradient! end

"""
    solve!(prob, optimizer; kwargs...)

Run the inversion with `optimizer`. Implemented by the Optim extension.
"""
function solve! end
