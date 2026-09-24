# =============================================================================
# Restart files: save the state of a `Simulation` to NetCDF and resume from it.
#
# A restart file holds the *state*, not the configuration: the user rebuilds the
# `Simulation` from the same script (domain, solid Earth, sea level, BCs) and
# passes `restart_from = path`. What is saved is everything a forward run
# mutates — the same set `snapshot!` captures (`sim.now`, the scalars of
# `sim.sealevel.bsl`, the clock) — plus the `ReferenceState`, which a restarted
# `Simulation` would otherwise rebuild from the BCs at the *restart* time, and
# the origin of the sparse-diagnostics clock.
#
# The variables are found by walking the structs rather than listed by hand, so a
# field added to `CurrentState` or `ReferenceState` is saved without touching this
# file, and a field of a type the walker does not know errors instead of being
# dropped. `io_dict` only lends its long names as attributes: its `map`s are unit
# conversions for display, and a restart must round-trip the raw values.
# =============================================================================

"""
$(TYPEDSIGNATURES)

Control the writing of restart files during a [`Simulation`](@ref).

# Fields
$(TYPEDFIELDS)

A restart file is written at each time of `t` and at the end of [`run!`](@ref). Each
write replaces the previous file, so `filename` always holds the latest state.
Pass it to `Simulation(...; restartout)`. A plain path is shorthand for a
`RestartOutput` with no intermediate times.

```jldoctest
julia> ro = RestartOutput("restart.nc"; t = [1f3, 2f3]);

julia> ro.t
2-element Vector{Float32}:
 1000.0
 2000.0
```
"""
mutable struct RestartOutput{T<:AbstractFloat}
    "the times at which a restart file is written, in addition to the end of `run!`"
    t::Vector{T}
    "the path of the restart file"
    filename::String
    "the index of the next time in `t`"
    k::Int
end

RestartOutput(filename::AbstractString; t = Float32[]) =
    RestartOutput(collect(t), String(filename), 1)

# Normalise the `restartout` keyword of `Simulation`: bring the times to the
# simulation's float type, sort them, and skip those not after the start of the run
# (a restarted simulation would otherwise overwrite the file it just read).
RestartOutput(::Nothing, ::Timer) = nothing
RestartOutput(filename::AbstractString, timer::Timer) =
    RestartOutput(RestartOutput(filename), timer)
function RestartOutput(ro::RestartOutput, timer::Timer{T}) where {T}
    t = sort(T.(ro.t))
    k = something(findfirst(>(timer.t_span[1]), t), length(t) + 1)
    return RestartOutput(t, ro.filename, k)
end

next_restart_time(::Nothing) = nothing
next_restart_time(ro::RestartOutput) = ro.k <= length(ro.t) ? ro.t[ro.k] : nothing

# Called by `advance_with_output!` when the integrator reaches the next restart time.
function restart_affect!(sim, progress = nothing)
    verbose_log(sim, progress) &&
        println("Writing restart file at simulation year $(sim.timer.t)...")
    write_restart(sim.restartout.filename, sim)
    sim.restartout.k += 1
    return nothing
end

const RESTART_TITLE = "FastIsostasy restart file"
const RESTART_VERSION = Int32(1)

# Every value a restart file holds, as `(name, owner, field)`: the value is
# `getfield(owner, field)` and `name` is its NetCDF variable, e.g.
# `now.kinematic.H_F_prev` or `bsl.z`.
function restart_leaves(sim::Simulation)
    leaves = Tuple{String,Any,Symbol}[]
    push_state_leaves!(leaves, "now", sim.now)
    push_state_leaves!(leaves, "ref", sim.ref)
    push_bsl_leaves!(leaves, "bsl", sim.sealevel.bsl)
    push!(leaves, ("timer.t", sim.timer, :t))
    push!(leaves, ("timer.t_sparse0", sim.timer, :t_sparse0))
    return leaves
end

function push_state_leaves!(leaves, prefix, obj)
    for f in fieldnames(typeof(obj))
        v = getfield(obj, f)
        name = "$prefix.$f"
        if v isa AbstractArray || v isa Real
            push!(leaves, (name, obj, f))
        elseif v isa ColumnAnomalies || v isa KinematicBSL
            push_state_leaves!(leaves, name, v)
        else
            error("Restart files cannot store `$name::$(typeof(v))`.")
        end
    end
    return leaves
end

# Mirrors `_copy_bsl!` (snapshot.jl): only the `Real` fields change during a run.
function push_bsl_leaves!(leaves, prefix, bsl)
    for f in fieldnames(typeof(bsl))
        v = getfield(bsl, f)
        name = "$prefix.$f"
        if v isa Real
            push!(leaves, (name, bsl, f))
        elseif v isa AbstractBSL
            push_bsl_leaves!(leaves, name, v)
        end
    end
    return leaves
end

# NetCDF has no Boolean type.
to_nc(x::AbstractArray{Bool}) = Int8.(x)
to_nc(x::AbstractArray) = x
to_nc(x::Bool) = Int8(x)
to_nc(x::Real) = x

restart_atts(f::Symbol) =
    haskey(io_dict, f) ? Dict{String,Any}("longname" => io_dict[f]["longname"]) :
    Dict{String,Any}()

function restart_dims(sz, xdim, ydim, extradims, name)
    nx, ny = Int(xdim.dimlen), Int(ydim.dimlen)
    if sz == (nx, ny)
        return [xdim, ydim]
    elseif length(sz) == 3 && sz[1:2] == (nx, ny)
        n = sz[3]
        return [xdim, ydim, get!(() -> NcDim("n$n", n), extradims, n)]
    end
    error("Restart files cannot store `$name` of size $sz on a $nx × $ny grid.")
end

"""
$(TYPEDSIGNATURES)

Write the current state of `sim` to the NetCDF restart file `filename`, replacing
any existing file. The file holds the state only: to resume, rebuild the
simulation with the same physics and pass `restart_from = filename` (see
[`Simulation`](@ref)), or call [`read_restart!`](@ref).

The values are stored at full precision on the full grid, with no unit
conversion or cropping. Arrays of size zero (the transient-creep branches of a
Maxwell mantle, or the kinematic buffers when using [`GoelzerBSLFormalism`](@ref)) are
not written.
"""
function write_restart(filename::AbstractString, sim::Simulation)
    domain = sim.domain
    xdim = NcDim("x", Array(domain.x), Dict("longname" => "x", "units" => "m"))
    ydim = NcDim("y", Array(domain.y), Dict("longname" => "y", "units" => "m"))
    scalardim = NcDim("scalar", 1)
    extradims = Dict{Int,NcDim}()

    vars = NcVar[]
    data = Pair{String,Any}[]
    for (name, owner, f) in restart_leaves(sim)
        v = getfield(owner, f)
        if v isa AbstractArray
            isempty(v) && continue
            dims = restart_dims(size(v), xdim, ydim, extradims, name)
            host = to_nc(Array(v))
        else
            dims = [scalardim]
            host = [to_nc(v)]
        end
        push!(vars, NcVar(name, dims; t = eltype(host), atts = restart_atts(f)))
        push!(data, name => host)
    end

    gatts = Dict{String,Any}(
        "title" => RESTART_TITLE,
        "restart_format_version" => RESTART_VERSION,
        "float_type" => string(eltype(domain.x)),
    )

    # Write next to the target and move it into place, so that a run killed
    # mid-write leaves the previous restart file intact.
    tmp = filename * ".tmp"
    isfile(tmp) && rm(tmp)
    NetCDF.create(tmp, vars; gatts = gatts) do nc
        for (name, host) in data
            NetCDF.putvar(nc, name, host)
        end
    end
    mv(tmp, filename; force = true)
    return filename
end

"""
$(TYPEDSIGNATURES)

Overwrite the state of `sim` with the one saved in the restart file `filename` by
[`write_restart`](@ref). This is what `Simulation(...; restart_from = filename)` calls.
Call it directly to resume a simulation driven by [`init_integrator`](@ref) and
[`step!`](@ref), before calling `init_integrator`.

`sim` must be set up with the physics of the run that wrote the file, and with
`t_span[1]` equal to the time at which the file was written. The grid, the time and
the sizes of all saved arrays are checked, and a mismatch is an error. Model
parameters (viscosity, lithospheric thickness, …) are not saved and therefore not
checked.
"""
function read_restart!(sim::Simulation, filename::AbstractString)
    isfile(filename) || error("Restart file `$filename` does not exist.")
    NetCDF.open(filename) do nc
        get(nc.gatts, "title", "") == RESTART_TITLE ||
            error("`$filename` is not a FastIsostasy restart file.")
        check_restart_axis(nc, "x", sim.domain.x, filename)
        check_restart_axis(nc, "y", sim.domain.y, filename)

        t_file = only(NetCDF.readvar(nc["timer.t"]))
        t0 = sim.timer.t_span[1]
        isapprox(t_file, t0; rtol = sqrt(eps(typeof(t0)))) || error(
            "`$filename` was written at t = $t_file, but the simulation starts at " *
            "t_span[1] = $t0. Start the restarted simulation at the time of the file.",
        )

        for (name, owner, f) in restart_leaves(sim)
            read_restart_leaf!(nc, name, owner, f)
        end
    end
    # Keep the clock on `t_span[1]` exactly, which the integrator starts from.
    sim.timer.t = sim.timer.t_span[1]
    return sim
end

function check_restart_axis(nc, name, axis, filename)
    file_axis = haskey(nc.vars, name) ? NetCDF.readvar(nc[name]) : nothing
    !isnothing(file_axis) && length(file_axis) == length(axis) &&
        isapprox(file_axis, Array(axis)) ||
        error("The $name-axis of `$filename` does not match the simulation domain.")
    return nothing
end

function read_restart_leaf!(nc, name, owner, f)
    dst = getfield(owner, f)
    if !haskey(nc.vars, name)
        dst isa AbstractArray && isempty(dst) && return nothing
        error(
            "The restart file lacks `$name`. Was it written by a simulation with " *
            "different physics (e.g. sea-level formalism or mantle rheology)?",
        )
    end
    src = NetCDF.readvar(nc[name])
    if dst isa AbstractArray
        size(src) == size(dst) || error(
            "`$name` has size $(size(src)) in the restart file but $(size(dst)) " *
            "in the simulation.",
        )
        copyto!(dst, convert(Array{eltype(dst)}, src))
    else
        val = convert(typeof(dst), only(src))
        if ismutable(owner)
            setfield!(owner, f, val)
        elseif val != dst
            # e.g. the scalars of the immutable `ReferenceState`
            error("`$name` is fixed at construction and differs from the restart file.")
        end
    end
    return nothing
end
