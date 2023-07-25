using Interpolations, NLsolve, NCDatasets, JLD2

"""

    load_bathymetry()

Load the bathymetry map from ETOPO-2022 on 1 minute arclength resolution.
"""
function load_bathymetry()
    # https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/60s/60s_bed_elev_netcdf/catalog.html?dataset=globalDatasetScan/ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc
    filename = "src/data/ETOPO_2022_v1_60s_N90W180_bed.nc"
    ds = NCDataset(filename, "r")
    lat = copy(ds["lat"][:,:])
    lon = copy(ds["lon"][:,:])
    bedrock_missing = copy(ds["z"][:,:])
    close(ds)

    bedrock = zeros(size(bedrock_missing))
    bedrock .= bedrock_missing

    return lat, lon, bedrock
end

"""

    get_cellsurface(lat::Vector{T}, lon::Vector{T}) where {T<:Float64}

Compute cell surface based on distortion generated by lat-lon projection.
"""
function get_cellsurface(lat::Vector{T}, lon::Vector{T}) where {T<:Float64}
    R = 6.371e6                     # Earth radius at equator (m)
    k = 1 ./ cos.( deg2rad.(lat) )
    dphi = mean([mean(diff(lat)), mean(diff(lon))])
    meridionallength_cell = deg2rad(dphi) * R
    azimutallength_cell = meridionallength_cell ./ k
    cellsurface = fill(meridionallength_cell, length(lon)) * azimutallength_cell'
    return cellsurface
end

"""

    gmsl_surface(bedrock::Matrix{T}, cellsurface::Matrix{T}, z::T) where {T<:Float64}

Compute the surface of ocean based on simple recognition of all cells with `bedrock < z`.
Scale by `cellsurface` to account for distortion by projection.
"""
function gmsl_surface(bedrock::Matrix{T}, cellsurface::Matrix{T}, z::T) where {T<:Float64}
    return sum( (bedrock .< z) .* cellsurface )
end

"""

    surface_over_depth(z_support::AbstractVector{T},
        bedrock::Matrix{T}, cellsurface::Matrix{T}) where {T<:Float64}

Return `A_support::Vector{T}`, the ocean surface evaluated at `z_support` based
on `bedrock` topography and distortion factor embedded in `cellsurface`.
"Support" refers to the fact that these vectors arwe then used to construct an
interpolator of ocean surface over depth.
"""
function surface_over_depth(z_support::AbstractVector{T},
    bedrock::Matrix{T}, cellsurface::Matrix{T}) where {T<:Float64}
    return [gmsl_surface(bedrock, cellsurface, z_support[i]) for i in eachindex(z_support)]
end


"""

    surfacechange_residual!(Vresidual::Vector, z::Vector, zk::T,
        A_itp::Interpolations.Extrapolation, deltaV::T) where {T<:Float64}

Return `A_support::Vector{T}`, the ocean surface evaluated at `z_support` based
on `bedrock` topography and distortion factor embedded in `cellsurface`.
The two former ones are then used as support points to construct an interpolator
of ocean surface over depth.
"""
function surfacechange_residual!(Vresidual::Vector, z::Vector, zk::T,
    A_itp::Interpolations.Extrapolation, deltaV::T) where {T<:Float64}
    Vresidual[1] = (z[1] - zk) * mean([A_itp(z[1]), A_itp(zk)]) - deltaV
end

"""

    discretize_oceansurface(; dz_support = 0.1, min_sle_anom = -150, max_sle_anom = 70)

Save a `.jld` file containing `z_support = min_sle_anom:dz_support:max_sle_anom` and
the ocean surface `A_support` corresponding to these GMSL.
"""
function discretize_oceansurface(;
    dz_support::Float64 = 0.1,
    min_sle_anom::Real = -150,
    max_sle_anom::Real = 70)
    lat, lon, bedrock = load_bathymetry()
    cellsurface = get_cellsurface(lat, lon)
    z_support = min_sle_anom:dz_support:max_sle_anom
    A_support = surface_over_depth(z_support, bedrock, cellsurface)
    jldsave("src/data/OceanSurface_dz=$(round(dz_support, digits=3))m.jld2";
        z_support, A_support)
end

# discretize_oceansurface(dz_support = 0.1)

"""

    OceanSurfaceChange(; z0 = 0.0)

Return a `mutable struct OceanSurfaceChange` containing:
 - `A_itp`, an interpolator of ocean surface over depth
 - `zk`, the current GMSL, optionally initialized with the keyword argument `z0`
 - `Ak`, the ocean surface associated with `zk`

An `osc::OceanSurfaceChange` can be used as function to update `osc.zk` and `osc.Ak`
based on `osc.A_itp` and an input `deltaV::Float64` by running:
```julia
osc(deltaV)
```
"""
mutable struct OceanSurfaceChange
    A_itp::Interpolations.Extrapolation
    zk::Float64
    Ak::Float64
end

function OceanSurfaceChange(; z0 = 0.0)
    discretized_oceansurface = jldopen("src/data/OceanSurface_dz=0.1m.jld2")
    z_support = discretized_oceansurface["z_support"]
    A_support = discretized_oceansurface["A_support"]
    A_itp = linear_interpolation(z_support, A_support)
    return OceanSurfaceChange(A_itp, z0, A_itp(z0))
end

function (osc::OceanSurfaceChange)(deltaV::Float64)
    scr!(Vresidual, z) = surfacechange_residual!(Vresidual, z,
        osc.zk, osc.A_itp, deltaV)
    
    mcp_opts = (reformulation = :smooth, autodiff = :forward, iterations = 100_000,
        ftol = 1e-5, xtol = 1e-5)
    if deltaV >= 0
        sol = mcpsolve(scr!, [osc.zk], [osc.zk+1.0], [osc.zk]; mcp_opts...)
    elseif deltaV < 0
        sol = mcpsolve(scr!, [osc.zk-1.0], [osc.zk], [osc.zk]; mcp_opts...)
    end
    

    if sol.f_converged || sol.x_converged
        osc.zk = sol.zero[1]
        osc.Ak = osc.A_itp(osc.zk)
        println("updated ocean surface :)")
    else
        error("NLsolve did not converge on ocean surface.")
    end
end

osc = OceanSurfaceChange()
A_pd = 3.625e14
pd_error = osc.Ak / A_pd
diff60 = osc.A_itp(60.0) / A_pd
diff140 = osc.A_itp(-140.0) / A_pd

V_antarctica = 26e6 * (1e3)^3
nV = 1000
dV = V_antarctica / nV
for i in 1:nV
    osc(dV)
end