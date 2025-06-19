################################################################################
# NetCDF output
################################################################################

io_dict = Dict{Symbol, Dict{String, String}}()
io_dict[:u] = Dict(
    "shortname" => "u",
    "longname" => "Viscous displacement",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:ue] = Dict(
    "shortname" => "ue",
    "longname" => "Elastic displacement",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:b] = Dict(
    "shortname" => "b",
    "longname" => "Bedrock elevation",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:z_ss] = Dict(
    "shortname" => "z_ss",
    "longname" => "Sea-surface height (SSH)",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:dudt] = Dict(
    "shortname" => "dudt",
    "longname" => "Viscous displacement rate",
    "units" => "m/yr",
    "dims" => "x y t",
)
io_dict[:dz_ss] = Dict(
    "shortname" => "dz_ss",
    "longname" => "SSH perturbation",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:maskgrounded] = Dict(
    "shortname" => "maskgrounded",
    "longname" => "Mask for grounded ice",
    "units" => "1",
    "dims" => "x y t",
)
io_dict[:H_ice] = Dict(
    "shortname" => "Hice",
    "longname" => "Ice thickness",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:H_water] = Dict(
    "shortname" => "Hwater",
    "longname" => "Water depth",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:u_x] = Dict(
    "shortname" => "u_x",
    "longname" => "Horizontal displacement in x",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:u_y] = Dict(
    "shortname" => "u_y",
    "longname" => "Horizontal displacement in y",
    "units" => "m",
    "dims" => "x y t",
)
io_dict[:bsl] = Dict(
    "shortname" => "bsl",
    "longname" => "Barystatic sea level",
    "units" => "m",
    "dims" => "t",
)

mutable struct NetcdfOutput{T<:AbstractFloat}
    t::Vector{T}
    filename::String
    buffer::Matrix{T}
    vars3D::Vector{Symbol}
    vars1D::Vector{Symbol}
    computation_time::T
end

function NetcdfOutput(Omega::ComputationDomain{T, L, M}, t, filename;
    vars3D = [:u, :ue, :b, :dz_ss],
    vars1D = [:bsl],
    Tout = Float32,
) where {T<:AbstractFloat, L, M}

    isfile(filename) && rm(filename)

    xatts = Dict("longname" => "x", "units" => "m")
    yatts = Dict("longname" => "y", "units" => "m")
    tatts = Dict("longname" => "time", "units" => "yr")

    xdim = NcDim("x", Omega.x, xatts)
    ydim = NcDim("y", Omega.y, yatts)
    tdim = NcDim("t", t, tatts)

    vars = NcVar[]
    for i in eachindex(vars3D)
        j = vars3D[i]
        varatts = Dict(
            "longname" => io_dict[j]["longname"],
            "units" => io_dict[j]["units"],
        )
        push!(vars,
            NcVar(io_dict[j]["shortname"], [xdim, ydim, tdim]; atts = varatts, t = Tout))
    end
    for i in eachindex(vars1D)
        j = vars1D[i]
        varatts = Dict(
            "longname" => io_dict[j]["longname"],
            "units" => io_dict[j]["units"],
        )
        push!(vars, NcVar(io_dict[j]["shortname"], [tdim]; atts = varatts, t = Tout))
    end

    if length(filename) > 0
        isfile(filename) && rm(filename)
        NetCDF.create(filename, vars) do nc
            nothing
        end
    end
    
    buffer = Matrix{Tout}(undef, Omega.nx, Omega.ny)
    return NetcdfOutput(Tout.(t), filename, buffer, vars3D, vars1D, Tout(0.0))
end

function select_preconfig(preconfig, intermediate, sparse)
    if preconfig == :intermediate
        return intermediate
    elseif preconfig == :sparse
        return sparse
    else
        error("Unknown preconfiguration")
    end
end

function write_nc!(ncout::NetcdfOutput{Tout}, state::CurrentState{T, M}, k::Int) where {
    T<:AbstractFloat, M<:KernelMatrix{T}, Tout<:AbstractFloat}
    for i in eachindex(ncout.vars3D)
        j = ncout.vars3D[i]
        if M == Matrix{T}
            ncout.buffer .= Tout.(getfield(state, ncout.vars3D[i]))
        else
            ncout.buffer .= Tout.(Array(getfield(state, ncout.vars3D[i])))
        end
        NetCDF.open(ncout.filename, mode = NC_WRITE) do nc
            NetCDF.putvar(nc, io_dict[j]["shortname"], ncout.buffer,
                start = [1, 1, k], count = [-1, -1, 1])
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

struct NativeOutput{T<:AbstractFloat}
    t::Vector{<:Real}
    vars::Vector{Symbol}
    vals::Dict{Symbol, Vector{Matrix{T}}}
end

function NativeOutput(; t = Float32[], vars = Symbol[], T = Float32)
    vals = Dict{Symbol, Vector{Matrix{T}}}()
    return NativeOutput(t, vars, vals)
end

function write_out!(nout::NativeOutput, now::CurrentState)
    for var in nout.vars
        push!(nout.vals[var], Array.(getfield(now, var)))
    end
end