# =============================================================================
# Observables and observations.
#
# An *observable* is a lightweight tag type that only says *which* model quantity
# is measured; it implements `observable_field(tag, sim)` returning the full 2-D
# scalar field the tag reads (materialized via a plain elementwise broadcast —
# backend-agnostic, Enzyme-legal, no scalar GPU reads).
#
# An *observation* (`Observation`) bundles a tag with the sampling locations
# (`points`), the times, the measured `data` and the noise `σ`. Several
# observations of possibly different tags are passed to an inversion as a vector.
#
# v1 samples on grid nodes (`CartesianIndex{2}`, the user-facing API). `extract!`
# reads from precomputed *linear* indices (backend-promoted `Vector{Int}` /
# `CuVector{Int}`, built once by the inversion constructor via
# `_obs_linear_indices` in problem.jl) via a `field[idx[i]]` gather, dual-path
# dispatched like `src/derivatives.jl`: a plain `@inbounds` loop on `Matrix`
# (CPU), a KA kernel otherwise (GPU) — scalar `CartesianIndex` reads on a
# `CuArray` are disallowed, so the gather must happen inside a kernel launch,
# not via one-index-at-a-time host code. Physical coordinates with
# differentiable bilinear interpolation are a later addition.
# =============================================================================

abstract type AbstractObservable end

"""
$(TYPEDSIGNATURES)

Total vertical bedrock displacement (viscous + elastic), `u + ue`.
"""
struct VerticalUpliftObservable <: AbstractObservable end

"""
$(TYPEDSIGNATURES)

Vertical bedrock uplift rate, `dudt` (viscous rate; the elastic contribution is
piecewise-constant between sparse diagnostic updates and is neglected here).
"""
struct VerticalUpliftRateObservable <: AbstractObservable end

"""
$(TYPEDSIGNATURES)

Relative sea level: sea-surface change minus bedrock uplift,
`(z_ss − z_ss_ref) − (u + ue)`.
"""
struct RelativeSeaLevelObservable <: AbstractObservable end

struct BarystaticContributionObservable <: AbstractObservable end

struct HorizontalDisplacementRateObservable <: AbstractObservable end


# --- per-tag full-field materialization (elementwise; backend-agnostic) ------

observable_field(::VerticalUpliftObservable, sim) = sim.now.u .+ sim.now.ue

observable_field(::VerticalUpliftRateObservable, sim) = sim.now.dudt

observable_field(::RelativeSeaLevelObservable, sim) = sim.now.z_ss .- (sim.now.u .+ sim.now.ue)

# observable_field(::BarystaticContributionObservable, sim) = sim.now.

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
    allunique(times) || throw(ArgumentError(
        "Observation times must be unique — a duplicate silently corrupts the " *
        "misfit (`_extract_plan`'s `findfirst` only ever matches the first " *
        "occurrence, so the duplicate's prediction slice stays zero)."))
    return Observation(tag, collect(points), collect(times), collect(data), σ)
end

nentries(obs::Observation) = length(obs.points) * length(obs.times)

# Slice of the flat prediction/data vector corresponding to time index `it`.
@inline function time_slice(obs::Observation, it::Int)
    np = length(obs.points)
    return (it - 1) * np + 1 : it * np
end

# --- points → backend-promoted linear indices ---------------------------------
#
# Shared by `Observation`-based inversions (`_linear_indices` in problem.jl) and
# `SimulatedObservable` below: `points` (`CartesianIndex{2}`, the user-facing API)
# converted once to linear indices and copied onto the same array family as
# `field` (`Matrix` on CPU, `CuMatrix` on GPU via `similar`), so `gather!`'s GPU
# branch never receives a host `Vector` alongside a device `field`.
function points_to_linear_indices(points, field)
    li = LinearIndices(size(field))
    idx_cpu = Int[li[p] for p in points]
    idx = similar(field, Int, length(idx_cpu))
    copyto!(idx, idx_cpu)
    return idx
end

# --- linear-index gather (dual path, mirrors src/derivatives.jl) -------------

@kernel function gather_kernel!(y, field, idx, offset::Int)
    i = @index(Global)
    y[offset + i] = field[idx[i]]
end

function gather!(y, field, idx, offset::Int)   # GPU / generic
    backend = get_backend(field)
    gather_kernel!(backend)(y, field, idx, offset; ndrange = length(idx))
    synchronize(backend)
    return nothing
end

function gather!(y::AbstractVector, field::Matrix, idx, offset::Int)   # CPU
    @inbounds for i in eachindex(idx)
        y[offset + i] = field[idx[i]]
    end
    return nothing
end

# Fill `pred[time_slice]` with the model values at the observation's (precomputed,
# backend-promoted linear) indices `lidx`, for the time whose index in `obs.times`
# is `it`. Assumes the sim is currently at that time.
function extract!(pred::AbstractVector, obs::Observation, it::Int, sim, lidx)
    sl = time_slice(obs, it)
    field = observable_field(obs.tag, sim)
    gather!(pred, field, lidx, first(sl) - 1)
    return nothing
end

# =============================================================================
# `SimulatedObservable` — forward-run virtual stations (roadmap §4c item 2).
#
# A lightweight station attached to a plain `Simulation`: records `tag` at grid
# `points` and `times` during `run!`/`step!` via the same `observable_field` +
# `gather!` dual path as `Observation`/`extract!` above, without ever writing a
# full 2-D field to output (unlike `NativeOutput`/`NetcdfOutput`). This is the
# forward-side counterpart of the lightweight extraction the inversion side
# already does in `forward_predict!` (problem.jl) — e.g. a handful of relative
# sea level time series instead of full fields saved for hindsight extraction.
# Entirely off the differentiated path: wired into `advance_with_output!`
# (integrators.jl) as a third output stream next to `nout`/`ncout`.
# =============================================================================

"""
    SimulatedObservable(tag, points, times, sim)

A virtual station recording `tag::AbstractObservable` at grid `points`
(`Vector{CartesianIndex{2}}`) and `times` during a forward run. `data` is flat,
points-fastest then times (`Observation`'s convention); `k` is the cursor index
into `times` of the next pending recording. Construct after `sim` exists (its
linear indices are backend-promoted onto `sim`'s field array family), then
attach with `attach_simobs!(sim, tag, points, times)`.
"""
mutable struct SimulatedObservable{O<:AbstractObservable, T<:AbstractFloat, LI, D}
    tag::O
    points::Vector{CartesianIndex{2}}
    times::Vector{T}
    linear_indices::LI
    data::D
    k::Int
end

function SimulatedObservable(tag::AbstractObservable, points, times, sim)
    field = observable_field(tag, sim)
    T = eltype(times)
    li = points_to_linear_indices(points, field)
    data = similar(field, T, length(points) * length(times))
    return SimulatedObservable(tag, collect(points), collect(times), li, data, 1)
end

nentries(so::SimulatedObservable) = length(so.points) * length(so.times)

@inline function time_slice(so::SimulatedObservable, it::Int)
    np = length(so.points)
    return (it - 1) * np + 1 : it * np
end

"""
    attach_simobs!(sim, tag, points, times) -> SimulatedObservable

Build a `SimulatedObservable` from `sim`'s current field layout and append it to
`sim.simobs`; `run!`/`step!` then record it automatically at each of `times`.
"""
function attach_simobs!(sim, tag::AbstractObservable, points, times)
    so = SimulatedObservable(tag, points, times, sim)
    push!(sim.simobs, so)
    return so
end

# Next pending recording time for one station, or `nothing` if exhausted.
next_simobs_time(so::SimulatedObservable) =
    so.k <= length(so.times) ? so.times[so.k] : nothing

# Record the current sim state into `so.data` at its pending time index and
# advance the cursor. Assumes the sim is currently at `next_simobs_time(so)`.
function record!(so::SimulatedObservable, sim)
    sl = time_slice(so, so.k)
    field = observable_field(so.tag, sim)
    gather!(so.data, field, so.linear_indices, first(sl) - 1)
    so.k += 1
    return nothing
end
