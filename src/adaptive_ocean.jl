"""
    OceanSurfaceChange(; z0 = 0.0)

Return a `mutable struct OceanSurfaceChange` containing:
 - `A_itp`: an interpolator of ocean surface over depth. Bias-free for present-day.
 - `A_pd`: the present-day ocean surface.
 - `A_k`: the ocean surface at time step `k`.
 - `z_k`: the GMSL at time step `k`.

An `osc::OceanSurfaceChange` can be used as function to update `osc.z_k` and `osc.A_k`
based on `osc.A_itp` and an input `delta_V::Float64` by running:

```julia
osc(delta_V)
```
"""
mutable struct OceanSurfaceChange{T<:AbstractFloat}
    z_k::T                  # sealevel at time step k
    A_k::T                  # ocean surface at time step k
    z::AbstractVector{T}    # support for pwlinear interpolator
    A::AbstractVector{T}    # values for pwlinear interpolator
    A_itp                   # pwlinear interpolator A(z) ::Interpolations.Extrapolation
    A_pd::T                 # present-day ocean surface
end

function OceanSurfaceChange(; A_ocean_pd::T = 3.625e14, z0::T = 0.0) where {T<:AbstractFloat}
    z, A, itp = load_oceansurfacefunction()
    A_itp(z) = A_ocean_pd / itp(0.0) * itp(z)
    return OceanSurfaceChange(z0, A_itp(z0), z, A, A_itp, A_ocean_pd)
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

        if sol.f_converged || sol.x_converged
            osc.z_k = sol.zero[1]
            osc.A_k = osc.A_itp(osc.z_k)
        else
            error("NLsolve did not converge on ocean surface.")
        end
    end
end


"""
    surfacechange_residual!(Vresidual::Vector, z::Vector, z_k::T,
        A_itp::Interpolations.Extrapolation, delta_V::T) where {T<:Float64}

Return `A_support::Vector{T}`, the ocean surface evaluated at `z_support` based
on `bedrock` topography and distortion factor embedded in `cellsurface`.
The two former ones are then used as support points to construct an interpolator
of ocean surface over depth.
"""
function surfacechange_residual!(Vresidual::Vector, z::Vector, z_k::T,
    A_itp, delta_V::T) where {T<:Float64}
    Vresidual[1] = (z[1] - z_k) * mean([A_itp(z[1]), A_itp(z_k)]) - delta_V
end