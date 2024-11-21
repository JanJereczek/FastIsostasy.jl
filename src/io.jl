################################################################################
# NetCDF output
################################################################################

mutable struct NetcdfOutput{T<:AbstractFloat}
    t::Vector{<:AbstractFloat}
    filename::String
    buffer::Matrix{T}
    varsfi3D::Vector{Symbol}
    varnames3D::Vector{String}
    varsfi1D::Vector{Symbol}
    varnames1D::Vector{String}
    computation_time::Float64
end

function NetcdfOutput(Omega::ComputationDomain{T, L, M}, t, filename;
    varsfi3D = interm_varsfi3D,
    varnames3D = interm_varnames3D,
    varlongnames3D = interm_varlongnames3D,
    varunits3D = interm_varunits3D,
    varsfi1D = interm_varsfi1D,
    varnames1D = interm_varnames1D,
    varlongnames1D = interm_varlongnames1D,
    varunits1D = interm_varunits1D,
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
    return NetcdfOutput(t, filename, buffer, varsfi3D, varnames3D,
        varsfi1D, varnames1D, 0.0)
end

interm_varsfi3D = [:u, :ue, :dudt, :b, :dz_ss, :z_ss, :maskgrounded, :H_ice,
    :H_water, :u_x, :u_y]
interm_varnames3D = ["u", "ue", "dudt", "b", "dz_ss", "z_ss",
    "maskgrounded", "Hice", "Hwater", "u_x", "u_y"]
interm_varlongnames3D = ["Viscous displacement", "Elastic displacement",
    "Viscous displacement rate", "Bedrock position", "SSH parturbation",
    "Sea-surface height (SSH)", "Mask for grounded ice", "Ice thickness", "Water depth",
    "Horizontal displacement in x", "Horizontal displacement in y"]
interm_varunits3D = ["m", "m", "m/yr", "m", "m", "m", "1", "m", "m", "m", "m"]

interm_varsfi1D = [:bsl]
interm_varnames1D = ["bsl"]
interm_varlongnames1D = ["Barystatic sea level"]
interm_varunits1D = ["m"]


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
    dudt::Vector{Matrix{T}}
    dz_ss::Vector{Matrix{T}}
end

function IntermediateOutput(Omega::ComputationDomain{T, L, M}, t_out::Vector{T}) where
    {T<:AbstractFloat, L, M<:KernelMatrix{T}}
    bsl = similar(t_out)
    placeholder = null(Omega)
    u = [copy(placeholder) for t in t_out]
    ue = [copy(placeholder) for t in t_out]
    dudt = [copy(placeholder) for t in t_out]
    dz_ss = [copy(placeholder) for t in t_out]
    return IntermediateOutput(t_out, T[], bsl, u, ue, dudt, dz_ss)
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
    out.dudt[k] .= copy(Array(now.dudt))
    out.dz_ss[k] .= copy(Array(now.dz_ss))
    return nothing
end
