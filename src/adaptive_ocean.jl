"""
    OceanSurfaceChange(; z0 = 0.0)

Return a `mutable struct OceanSurfaceChange` containing:
- `z_k`: the GMSL at current time step `k`.
- `A_k`: the ocean surface at current time step `k`.
- `z`: a vector of GMSL values used as knots for interpolation.
- `A`: a vector of ocean surface values used as knots for interpolation.
- `A_itp`: an interpolator of ocean surface over depth. Bias-free for present-day.
- `A_pd`: the present-day ocean surface.
- `res`: residual of the nonlinear equation solved numerically.

An `osc::OceanSurfaceChange` can be used as function to update `osc.z_k` and `osc.A_k`
based on `osc.A_itp` and an input `delta_V` by running:

```julia
osc(delta_V)
```
"""
mutable struct OceanSurfaceChange{T<:AbstractFloat}
    z_k::T
    A_k::T
    z::Vector{T}
    A::Vector{T}
    A_itp::Any
    A_pd::T
    residual::T
end
# ::{T, 1, Interpolations.Extrapolation{T, 1, Interpolations.GriddedInterpolation{T, 1, Vector{T},
# Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, Throw{Nothing}}}

function OceanSurfaceChange(; T = Float64, A_ocean_pd = T(3.625e14), z0 = T(0.0))
    z, A, itp = load_oceansurfacefunction()
    A_itp(z) = A_ocean_pd / itp(T(0.0)) * itp(z)
    return OceanSurfaceChange(z0, A_itp(z0), z, A, A_itp, A_ocean_pd, T(1e10))
end

function (osc::OceanSurfaceChange{T})(delta_V::T) where {T<:AbstractFloat}
    scr!(Vresidual, z) = surfacechange_residual!(Vresidual, z, osc.z_k, osc.A_itp,
        delta_V)
    mcp_opts = (reformulation = :smooth, autodiff = :forward, iterations = 100_000,
        ftol = 1e-5, xtol = 1e-5)

    # Update ocean surface within reasonable bounds defined by z_max_update and
    # only if sea-level contribution is nonzero.
    if delta_V != 0
        if delta_V > 0
            sol = mcpsolve(scr!, [osc.z_k], [maximum(osc.z)], [osc.z_k]; mcp_opts...)
        elseif delta_V < 0
            sol = mcpsolve(scr!, [minimum(osc.z)], [osc.z_k], [osc.z_k]; mcp_opts...)
        end

        surfacechange_residual!(osc, sol.zero[1], delta_V)

        # Residual must be less than 10 μm sea level in piecewise linear approximation.
        # Otherwise, use piecewise constant approximation = very rare exception.
        if osc.residual < 1e-5 * osc.A_pd
            osc.z_k = sol.zero[1]
            osc.A_k = osc.A_itp(osc.z_k)
        else
            osc.z_k += delta_V / osc.A_itp(osc.z_k)
            osc.A_k = osc.A_itp(osc.z_k)
        end
    end
end


"""
    surfacechange_residual!(osc, z, delta_V)
    surfacechange_residual!(Vresidual, z, z_k, A_itp, delta_V)

Update the residual of the piecewise linear approximation, used to solve the
sea-level/ocean-surface nonlinearity.
"""
function surfacechange_residual!(Vresidual::Vector, z::Vector, z_k::T,
    A_itp, delta_V::T) where {T<:AbstractFloat}
    Vresidual[1] = (z[1] - z_k) * mean([A_itp(z[1]), A_itp(z_k)]) - delta_V
end

function surfacechange_residual!(osc::OceanSurfaceChange{T}, z::T, delta_V::T) where
    {T<:AbstractFloat}
    osc.residual = (z - osc.z_k) * mean([osc.A_itp(z), osc.A_itp(osc.z_k)]) - delta_V
    return nothing
end