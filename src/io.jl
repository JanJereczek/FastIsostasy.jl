################################################################################
# NetCDF output
################################################################################

mutable struct NetcdfOutput{T<:AbstractFloat}
    t::Vector{T}
    filename::String
    buffer::Matrix{T}
    varsfi3D::Vector{Symbol}
    varnames3D::Vector{String}
    varsfi1D::Vector{Symbol}
    varnames1D::Vector{String}
    computation_time::T
end

function NetcdfOutput(Omega::ComputationDomain{T, L, M}, t, filename, preconfig;
    varsfi3D = select_preconfig(preconfig, interm_varsfi3D, sparse_varsfi3D),
    varnames3D = select_preconfig(preconfig, interm_varnames3D, sparse_varnames3D),
    varlongnames3D = select_preconfig(preconfig, interm_varlongnames3D, sparse_varlongnames3D),
    varunits3D = select_preconfig(preconfig, interm_varunits3D, sparse_varunits3D),
    varsfi1D = select_preconfig(preconfig, interm_varsfi1D, sparse_varsfi1D),
    varnames1D = select_preconfig(preconfig, interm_varnames1D, sparse_varnames1D),
    varlongnames1D = select_preconfig(preconfig, interm_varlongnames1D, sparse_varlongnames1D),
    varunits1D = select_preconfig(preconfig, interm_varunits1D, sparse_varunits1D),
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
    # @show varsfi3D, varnames3D, varlongnames3D, varunits3D, varsfi1D, varnames1D, varlongnames1D, varunits1D
    for i in eachindex(varnames3D)
        varatts = Dict("longname" => varlongnames3D[i], "units" => varunits3D[i])
        push!(vars, NcVar(varnames3D[i], [xdim, ydim, tdim]; atts = varatts, t = Tout))
    end
    for i in eachindex(varnames1D)
        varatts = Dict("longname" => varlongnames1D[i], "units" => varunits1D[i])
        push!(vars, NcVar(varnames1D[i], [tdim]; atts = varatts, t = Tout))
    end

    if length(filename) > 0
        isfile(filename) && rm(filename)
        NetCDF.create(filename, vars) do nc
            nothing
        end
    end
    
    buffer = Matrix{Tout}(undef, Omega.Nx, Omega.Ny)
    return NetcdfOutput(Tout.(t), filename, buffer, varsfi3D, varnames3D,
        varsfi1D, varnames1D, Tout(0.0))
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

const sparse_varsfi3D = [:u, :ue, :b, :z_ss]
const interm_varsfi3D = vcat(sparse_varsfi3D, [:dudt, :dz_ss, :maskgrounded, :H_ice,
    :H_water, :u_x, :u_y])

const sparse_varnames3D = ["u", "ue", "b", "z_ss"]
const interm_varnames3D = vcat(sparse_varnames3D, ["dudt", "dz_ss",
    "maskgrounded", "Hice", "Hwater", "u_x", "u_y"])

const sparse_varlongnames3D = ["Viscous displacement", "Elastic displacement",
    "Bedrock elevation", "Sea-surface height (SSH)"]
const interm_varlongnames3D = vcat(sparse_varlongnames3D, 
    ["Viscous displacement rate", "SSH perturbation",
    "Mask for grounded ice", "Ice thickness", "Water depth",
    "Horizontal displacement in x", "Horizontal displacement in y"])

const sparse_varunits3D = ["m", "m", "m", "m"]
const interm_varunits3D = vcat(sparse_varunits3D, ["m/yr", "m", "1", "m",
    "m", "m", "m"])

const sparse_varsfi1D = [:bsl]
const sparse_varnames1D = ["bsl"]
const sparse_varlongnames1D = ["Barystatic sea level"]
const sparse_varunits1D = ["m"]

const interm_varsfi1D = sparse_varsfi1D
const interm_varnames1D = sparse_varnames1D
const interm_varlongnames1D = sparse_varlongnames1D
const interm_varunits1D = sparse_varunits1D

function write_nc!(ncout::NetcdfOutput{Tout}, state::CurrentState{T, M}, k::Int) where {
    T<:AbstractFloat, M<:KernelMatrix{T}, Tout<:AbstractFloat}
    for i in eachindex(ncout.varnames3D)
        if M == Matrix{T}
            ncout.buffer .= Tout.(getfield(state, ncout.varsfi3D[i]))
        else
            ncout.buffer .= Tout.(Array(getfield(state, ncout.varsfi3D[i])))
        end
        NetCDF.open(ncout.filename, mode = NC_WRITE) do nc
            NetCDF.putvar(nc, ncout.varnames3D[i], ncout.buffer,
                start = [1, 1, k], count = [-1, -1, 1])
        end
    end
    for i in eachindex(ncout.varnames1D)
        val = Tout(getfield(state, ncout.varsfi1D[i]))
        NetCDF.open(ncout.filename, mode = NC_WRITE) do nc
            NetCDF.putvar(nc, ncout.varnames1D[i], [val], start = [k], count = [1])
        end
    end
end

################################################################################
# Postprocessing output (only used for optimization)
################################################################################

abstract type Output end
struct MinimalOutput{T<:AbstractFloat} <: Output
    t::Vector{T}
    t_ode::Vector{T}
end

mutable struct SparseOutput{T<:AbstractFloat} <: Output
    t::Vector{T}
    t_ode::Vector{T}
    u::Vector{Matrix{T}}
    ue::Vector{Matrix{T}}
end

function SparseOutput(Omega::ComputationDomain{T, L, M}, t_out::Vector{T}) where
    {T<:AbstractFloat, L, M<:KernelMatrix{T}}
    # initialize with placeholders
    placeholder = Array(null(Omega))
    u = [copy(placeholder) for t in t_out]
    ue = [copy(placeholder) for t in t_out]
    return SparseOutput(t_out, T[], u, ue)
end

mutable struct IntermediateOutput{T<:AbstractFloat} <: Output
    t::Vector{T}
    t_ode::Vector{T}
    bsl::Vector{T}
    u::Vector{Matrix{T}}
    ue::Vector{Matrix{T}}
    u_x::Vector{Matrix{T}}
    u_y::Vector{Matrix{T}}
    dudt::Vector{Matrix{T}}
    dz_ss::Vector{Matrix{T}}
end

function IntermediateOutput(Omega::ComputationDomain{T, L, M}, t_out::Vector{T}) where
    {T<:AbstractFloat, L, M<:KernelMatrix{T}}
    bsl = similar(t_out)
    placeholder = null(Omega)
    u = [copy(placeholder) for t in t_out]
    ue = [copy(placeholder) for t in t_out]
    u_x = [copy(placeholder) for t in t_out]
    u_y = [copy(placeholder) for t in t_out]
    dudt = [copy(placeholder) for t in t_out]
    dz_ss = [copy(placeholder) for t in t_out]
    return IntermediateOutput(t_out, T[], bsl, u, ue, u_x, u_y, dudt, dz_ss)
end

"""
    write_out!(now, out, k)

Write results in output vectors.
"""
function write_out!(out::SparseOutput{T}, now::CurrentState{T, M}, k::Int) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    out.u[k] .= copy(Array(now.u))
    out.ue[k] .= copy(Array(now.ue))
    return nothing
end

function write_out!(out::IntermediateOutput{T}, now::CurrentState{T, M}, k::Int) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    out.bsl[k] = now.bsl
    out.u[k] .= copy(Array(now.u))
    out.ue[k] .= copy(Array(now.ue))
    out.u_x[k] .= copy(Array(now.u_x))
    out.u_y[k] .= copy(Array(now.u_y))
    out.dudt[k] .= copy(Array(now.dudt))
    out.dz_ss[k] .= copy(Array(now.dz_ss))
    return nothing
end
