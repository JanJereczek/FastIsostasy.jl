"""
    get_cellsurface(lat::Vector{T}, lon::Vector{T}) where {T<:Float64}

Compute cell surface based on distortion generated by lat-lon projection.
"""
function get_cellsurface(lat::Vector{T}, lon::Vector{T}) where {T<:Float64}
    R = 6371e3                     # Earth radius at equator (m)
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
    A_itp, deltaV::T) where {T<:Float64}
    Vresidual[1] = (z[1] - zk) * mean([A_itp(z[1]), A_itp(zk)]) - deltaV
end


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
    A_itp#::Interpolations.Extrapolation
    A_pd::Real
    zk::Float64
    Ak::Float64
end

function OceanSurfaceChange(; A_ocean_pd = 3.625e14, z0 = 0.0)
    z, A, itp = load_oceansurfacefunction()
    A_itp(z) = A_ocean_pd / itp(0.0) * itp(z)
    return OceanSurfaceChange(A_itp, A_ocean_pd, z0, A_itp(z0))
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
        # println("updated ocean surface :)")
    else
        error("NLsolve did not converge on ocean surface.")
    end
end

# diff60 = osc.A_itp(60.0) / A_pd
# diff140 = osc.A_itp(-140.0) / A_pd

# V_antarctica = 26e6 * (1e3)^3
# nV = 1000
# dV = V_antarctica / nV
# for i in 1:nV
#     osc(dV)
# end