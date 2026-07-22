"""
$(TYPEDSIGNATURES)

Define a cropping strategy for the output of the simulation.

# Available subtypes
- [`PaddedOutputCrop`](@ref)
- [`AsymetricOutputCrop`](@ref)
"""
abstract type AbstractOutputCrop end

"""
$(TYPEDSIGNATURES)

Define a symmetric cropping strategy for the output of the simulation.
The output will be cropped by `pad` elements on each side of the domain.

# Fields
- `pad`: number of elements to crop on each side of the domain.
"""
struct PaddedOutputCrop <: AbstractOutputCrop
    pad::Int
end

"""
$(TYPEDSIGNATURES)

Define an asymmetric cropping strategy for the output of the simulation.

# Fields
- `pad_x1`: number of elements to crop on the left side of the domain.
- `pad_x2`: number of elements to crop on the right side of the domain.
- `pad_y1`: number of elements to crop on the bottom side of the domain.
- `pad_y2`: number of elements to crop on the top side of the domain.
"""
struct AsymetricOutputCrop <: AbstractOutputCrop
    pad_x1::Int
    pad_x2::Int
    pad_y1::Int
    pad_y2::Int
end

"""
$(TYPEDSIGNATURES)

Crop a vector using the specified cropping strategy.
"""
crop(x::V, c::PaddedOutputCrop) where {V<:AbstractVector} = x[(c.pad+1):(end-c.pad)]
crop(X::M, c::PaddedOutputCrop) where {M<:AbstractMatrix} =
    X[(c.pad+1):(end-c.pad), (c.pad+1):(end-c.pad)]

function crop(x::V, c::AsymetricOutputCrop) where {V<:AbstractVector}
    return x[(c.pad_x1+1):(end-c.pad_x2)]
end

function crop(X::M, c::AsymetricOutputCrop) where {M<:AbstractMatrix}
    return X[(c.pad_x1+1):(end-c.pad_x2), (c.pad_y1+1):(end-c.pad_y2)]
end

function crop_promote!(out, state, var, Tout, M, oc)
    if M isa Matrix
        out .= Tout.(crop(getfield(state, var), oc))
    else
        out .= Tout.(crop(Array(getfield(state, var)), oc))
    end
end

################################################################################
# NetCDF output
################################################################################

io_dict = Dict{Symbol,Dict{String,Any}}()
io_dict[:u] = Dict(
    "shortname" => "u",
    "longname" => "Viscous displacement",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:ue] = Dict(
    "shortname" => "ue",
    "longname" => "Elastic displacement",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:u_x] = Dict(
    "shortname" => "u_x",
    "longname" => "Horizontal displacement in x",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:u_y] = Dict(
    "shortname" => "u_y",
    "longname" => "Horizontal displacement in y",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:dudt] = Dict(
    "shortname" => "dudt",
    "longname" => "Viscous displacement rate",
    "units" => "mm/yr",
    "dims" => "x y t",
    "map" => x -> 1.0f3 .* x,     # Convert from m/yr to mm/yr
)
io_dict[:u_eq] = Dict(
    "shortname" => "u_eq",
    "longname" => "Equilibrium displacement",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:H_ice] = Dict(
    "shortname" => "Hice",
    "longname" => "Ice thickness",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:H_af] = Dict(
    "shortname" => "Haf",
    "longname" => "Ice thickness above floatation",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:H_water] = Dict(
    "shortname" => "Hwater",
    "longname" => "Water depth",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:z_b] = Dict(
    "shortname" => "z_b",
    "longname" => "Bedrock position",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:z_ss] = Dict(
    "shortname" => "z_ss",
    "longname" => "Sea-surface height (SSH)",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:dz_ss] = Dict(
    "shortname" => "dz_ss",
    "longname" => "SSH perturbation",
    "units" => "m",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:maskgrounded] = Dict(
    "shortname" => "maskgrounded",
    "longname" => "Mask for grounded ice",
    "units" => "1",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:maskocean] = Dict(
    "shortname" => "maskocean",
    "longname" => "Mask for ocean",
    "units" => "1",
    "dims" => "x y t",
    "map" => x -> x,
)
io_dict[:z_bsl] = Dict(
    "shortname" => "z_bsl",
    "longname" => "Barystatic sea level",
    "units" => "m",
    "dims" => "t",
    "map" => x -> x,
)
io_dict[:effective_viscosity] = Dict(
    "shortname" => "effective_viscosity",
    "longname" => "Effective viscosity",
    "units" => "Pa s",
    "dims" => "x y",
    "map" => x -> log10(x),
)
io_dict[:tau] = Dict(
    "shortname" => "tau",
    "longname" => "Relaxation time",
    "units" => "yr",
    "dims" => "x y",
    "map" => x -> x,
)
io_dict[:litho_thickness] = Dict(
    "shortname" => "litho_thickness",
    "longname" => "Lithosphere thickness",
    "units" => "km",
    "dims" => "x y",
    "map" => x -> 1.0f-3 .* x,        # Convert from m to km
)
io_dict[:pseudodiff_scaling] = Dict(
    "shortname" => "R",
    "longname" => "Pseudo-differential scaling",
    "units" => "1",
    "dims" => "x y",
    "map" => x -> x,
)
io_dict[:maskactive] = Dict(
    "shortname" => "active_mask",
    "longname" => "Mask where load is active",
    "units" => "1",
    "dims" => "x y",
    "map" => x -> x,
)
# io_dict[:green_elastic] = Dict(
#     "shortname" => "green_elastic",
#     "longname" => "Elastic Green's function",
#     "units" => "?",
#     "dims" => "x y",
#     "map" => x -> x,
# )
# io_dict[:green_geoid] = Dict(
#     "shortname" => "green_geoid",
#     "longname" => "Geoid Green's function",
#     "units" => "?",
#     "dims" => "x y",
#     "map" => x -> x,
# )

"""
$(TYPEDSIGNATURES)

Define the output NetCDF file for the simulation.

# Fields
- `t`: a vector of time points at which the variable is defined.
- `filename`: the name of the NetCDF file.
- `buffer`: a buffer to store the output data before writing to the NetCDF file.
- `vars3D`: a vector of symbols representing the 3D variables to be output.
- `vars1D`: a vector of symbols representing the 1D variables to be output.
- `params2D`: a vector of symbols representing the 2D parameters to be output.
- `oc`: an instance of [`AbstractOutputCrop`](@ref) that defines the cropping strategy for the output.
- `k`: the current time step index for writing to the NetCDF file.
"""
mutable struct NetcdfOutput{
    T<:AbstractFloat,
    OC,                 # <: AbstractOutputCrop
}
    t::Vector{T}
    filename::String
    buffer::Matrix{T}
    vars3D::Vector{Symbol}
    vars1D::Vector{Symbol}
    params2D::Vector{Symbol}
    oc::OC
    k::Int
end

function NetcdfOutput(
    domain::RegionalDomain{T,L,M},
    t,
    filename;
    vars3D = [:u, :ue, :z_b, :dz_ss],
    vars1D = [:z_bsl],
    params2D = [:effective_viscosity],
    Tout = Float32,
    output_crop = PaddedOutputCrop(0),
    solidearth = nothing,
) where {T<:AbstractFloat,L,M}

    isfile(filename) && rm(filename)

    xatts = Dict("longname" => "x", "units" => "m")
    yatts = Dict("longname" => "y", "units" => "m")
    tatts = Dict("longname" => "time", "units" => "yr")

    xdim = NcDim("x", crop(domain.x, output_crop), xatts)
    ydim = NcDim("y", crop(domain.y, output_crop), yatts)
    tdim = NcDim("t", t, tatts)
    buffer = crop(Matrix{Tout}(undef, domain.nx, domain.ny), output_crop)

    vars = NcVar[]
    for i in eachindex(vars3D)
        j = vars3D[i]
        varatts =
            Dict("longname" => io_dict[j]["longname"], "units" => io_dict[j]["units"])
        push!(
            vars,
            NcVar(
                io_dict[j]["shortname"],
                [xdim, ydim, tdim];
                atts = varatts,
                t = Tout,
            ),
        )
    end
    for i in eachindex(vars1D)
        j = vars1D[i]
        varatts =
            Dict("longname" => io_dict[j]["longname"], "units" => io_dict[j]["units"])
        push!(vars, NcVar(io_dict[j]["shortname"], [tdim]; atts = varatts, t = Tout))
    end
    for i in eachindex(params2D)
        j = params2D[i]
        varatts =
            Dict("longname" => io_dict[j]["longname"], "units" => io_dict[j]["units"])
        push!(
            vars,
            NcVar(io_dict[j]["shortname"], [xdim, ydim]; atts = varatts, t = Tout),
        )
    end

    if occursin(".nc", filename)
        isfile(filename) && rm(filename)
        NetCDF.create(filename, vars) do nc
            nothing
        end

        if solidearth !== nothing
            for i in eachindex(params2D)
                j = params2D[i]
                crop_promote!(buffer, solidearth, j, Tout, M, output_crop)
                NetCDF.open(filename, mode = NC_WRITE) do nc
                    NetCDF.putvar(
                        nc,
                        io_dict[j]["shortname"],
                        io_dict[j]["map"].(buffer),
                    )
                end
            end
        end

        println("NetCDF file $filename was created correctly.")
    else
        @warn "NetCDF filename does not end with '.nc' and is therefore ignored."
    end

    return NetcdfOutput(
        Tout.(t),
        filename,
        buffer,
        vars3D,
        vars1D,
        params2D,
        output_crop,
        1,
    )
end

function write_nc!(
    ncout::NetcdfOutput{Tout},
    state::CurrentState{T,M},
    k::Int,
) where {T<:AbstractFloat,M,Tout<:AbstractFloat}
    for i in eachindex(ncout.vars3D)
        j = ncout.vars3D[i]
        crop_promote!(ncout.buffer, state, ncout.vars3D[i], Tout, M, ncout.oc)
        NetCDF.open(ncout.filename, mode = NC_WRITE) do nc
            NetCDF.putvar(
                nc,
                io_dict[j]["shortname"],
                ncout.buffer,
                start = [1, 1, k],
                count = [-1, -1, 1],
            )
        end
    end
    for i in eachindex(ncout.vars1D)
        j = ncout.vars1D[i]
        val = Tout(getfield(state, j))
        NetCDF.open(ncout.filename, mode = NC_WRITE) do nc
            NetCDF.putvar(nc, io_dict[j]["shortname"], [val], start = [k], count = [1])
        end
    end
end

################################################################################
# Postprocessing output (only used for optimization)
################################################################################

"""
$(TYPEDSIGNATURES)

Define the native output which will be updated over the simulation.

Initialization example:
```julia
nout = NativeOutput(vars = [:u, :ue, :b, :dz_ss, :H_ice, :H_water, :u_x, :u_y],
    t = collect(0:1f3:10f3))
```

# Fields
- `t`: a vector of time points at which the variable is defined.
- `vars`: a vector of symbols representing the variables to be output.
- `vals`: a dictionary mapping each variable symbol to a vector of matrices containing the output data.
- `computation_time`: the total computation time for the simulation.
- `k`: the current time step index for writing to the output.
"""
mutable struct NativeOutput{T<:AbstractFloat}
    t::Vector{T}
    vars::Vector{Symbol}
    vals::Dict{Symbol,Vector{Matrix{T}}}
    computation_time::T
    k::Int
end

function NativeOutput(; t = Float32[], vars = Symbol[], T = Float32)
    vals = Dict{Symbol,Vector{Matrix{T}}}(var => Matrix{T}[] for var in vars)
    return NativeOutput(t, vars, vals, T(0), 1)
end

function write_out!(nout::NativeOutput, now::CurrentState)
    for var in nout.vars
        push!(nout.vals[var], Array(getfield(now, var)))
    end
end