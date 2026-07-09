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

# --- per-tag full-field materialization (elementwise; backend-agnostic) ------

observable_field(::VerticalUpliftObservable, sim) = sim.now.u .+ sim.now.ue

observable_field(::VerticalUpliftRateObservable, sim) = sim.now.dudt

observable_field(::RelativeSeaLevelObservable, sim) =
    (sim.now.z_ss .- sim.ref.z_ss) .- (sim.now.u .+ sim.now.ue)

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
