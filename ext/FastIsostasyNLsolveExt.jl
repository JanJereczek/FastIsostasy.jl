using NLsolve: mcpsolve

function update_ocean!(os::PiecewiseLinearOceanSurface{T}, delta_V::T) where {T<:AbstractFloat}
    scr!(Vresidual, z) = surfacechange_residual!(Vresidual, z, os.z_k, os.A_itp, delta_V)
    mcp_opts = (reformulation = :smooth, autodiff = :forward, iterations = 100_000,
        ftol = 1e-5, xtol = 1e-5)

    if delta_V != 0
        os.z_k += delta_V / os.A_itp(os.z_k)
        os.A_k = os.A_itp(os.z_k)
    end
    return nothing
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