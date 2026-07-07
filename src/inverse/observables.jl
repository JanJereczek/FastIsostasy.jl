# =============================================================================
# Observables and observations.
#
# An *observable* is a lightweight tag type that only says *which* model quantity
# is measured; it implements `observable_value(tag, sim, idx)` returning the
# scalar model prediction at grid index `idx` (a `CartesianIndex{2}`) at the
# current simulation time.
#
# An *observation* (`Observation`) bundles a tag with the sampling locations
# (`points`), the times, the measured `data` and the noise `σ`. Several
# observations of possibly different tags are passed to an inversion as a vector.
#
# v1 samples on grid nodes (`CartesianIndex{2}`). Physical coordinates with
# differentiable bilinear interpolation are a later addition.
# =============================================================================

abstract type AbstractObservable end

"""
    VerticalUpliftObservable()

Total vertical bedrock displacement (viscous + elastic), `u + ue`.
"""
struct VerticalUpliftObservable <: AbstractObservable end

"""
    VerticalUpliftRateObservable()

Vertical bedrock uplift rate, `dudt` (viscous rate; the elastic contribution is
piecewise-constant between sparse diagnostic updates and is neglected here).
"""
struct VerticalUpliftRateObservable <: AbstractObservable end

"""
    RelativeSeaLevelObservable()

Relative sea level: sea-surface change minus bedrock uplift,
`(z_ss − z_ss_ref) − (u + ue)`.
"""
struct RelativeSeaLevelObservable <: AbstractObservable end

# `HorizontalDisplacementRateObservable` is intentionally left undefined for now.

# --- per-index model predictions (scalar reads; CPU grid-index sampling) ------

@inline observable_value(::VerticalUpliftObservable, sim, idx) =
    sim.now.u[idx] + sim.now.ue[idx]

@inline observable_value(::VerticalUpliftRateObservable, sim, idx) =
    sim.now.dudt[idx]

@inline observable_value(::RelativeSeaLevelObservable, sim, idx) =
    (sim.now.z_ss[idx] - sim.ref.z_ss[idx]) - (sim.now.u[idx] + sim.now.ue[idx])

"""
    Observation(tag, points, times, data; σ = 1)

A set of measurements of `tag::AbstractObservable` at grid `points`
(`Vector{CartesianIndex{2}}`) and `times`. `data` is ordered points-fastest,
then times: `[p1@t1, p2@t1, …, p1@t2, …]`, i.e. length `npoints * ntimes`.
`σ` is a scalar or a per-entry vector of the same length.
"""
struct Observation{O<:AbstractObservable, T<:AbstractFloat, S}
    tag::O
    points::Vector{CartesianIndex{2}}
    times::Vector{T}
    data::Vector{T}
    σ::S
end

function Observation(tag, points, times, data; σ = one(eltype(data)))
    np, nt = length(points), length(times)
    length(data) == np * nt || throw(DimensionMismatch(
        "data has length $(length(data)) but expected npoints*ntimes = $(np*nt)."))
    σ isa AbstractVector && length(σ) != np * nt && throw(DimensionMismatch(
        "per-entry σ must have length npoints*ntimes = $(np*nt)."))
    return Observation(tag, collect(points), collect(times), collect(data), σ)
end

nentries(obs::Observation) = length(obs.points) * length(obs.times)

# Slice of the flat prediction/data vector corresponding to time index `it`.
@inline function time_slice(obs::Observation, it::Int)
    np = length(obs.points)
    return (it - 1) * np + 1 : it * np
end

# Fill `pred[time_slice]` with the model values at `obs.points` for the time whose
# index in `obs.times` is `it`. Assumes the sim is currently at that time.
function extract!(pred::AbstractVector, obs::Observation, it::Int, sim)
    sl = time_slice(obs, it)
    @inbounds for (k, idx) in enumerate(obs.points)
        pred[sl[k]] = observable_value(obs.tag, sim, idx)
    end
    return nothing
end
