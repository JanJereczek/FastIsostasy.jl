using NLsolve: mcpsolve

###########################################################################################
# PiecewiseLinearOceanSurface
###########################################################################################

"""
    PiecewiseLinearOceanSurface{T}
    PiecewiseLinearOceanSurface(; ref, mcp_opts)

A `mutable struct` that is only available if `using NLsolve` and contains:
- `ref`: a [`ReferenceOcean`](@ref).
- `z`: the current BSL.
- `A`: the current ocean surface.
- `residual`: residual of the nonlinear equation solved numerically.
- `mcp_opts`: options for the MCP solver, such as `reformulation`, `autodiff`, `iterations`, `ftol`, and `xtol`.

Note that, unlike [`ConstantOceanSurface`](@ref) and [`PiecewiseConstantOceanSurface`](@ref), this will only work if `using NLsolve`.
"""
mutable struct PiecewiseLinearOceanSurface{T, R<:ReferenceOcean{T}} <: AbstractOceanSurface{T}
    ref::R
    z::T
    A::T
    residual::T
    mcp_opts::NamedTuple
end

function PiecewiseLinearOceanSurface(; ref = ReferenceOcean(),
    mcp_opts = (reformulation = :smooth, autodiff = :forward,
        iterations = 100_000, ftol = 1e-5, xtol = 1e-5) )
    return PiecewiseLinearOceanSurface(ref, z, A, typemax(eltype(ref)), mcp_opts)
end

function update_ocean!(os::PiecewiseLinearOceanSurface, delta_V)
    scr!(Vresidual, z) = surfacechange_residual!(Vresidual, z, os.z, os.ref.A_itp, delta_V)

    # Update ocean surface within reasonable bounds defined by z_max_update and
    # only if sea-level contribution is nonzero.
    if delta_V != 0
        if delta_V > 0
            sol = mcpsolve(scr!, [os.z], [maximum(os.ref.z_vec)], [os.z]; os.mcp_opts...)
        elseif delta_V < 0
            sol = mcpsolve(scr!, [minimum(os.ref.z_vec)], [os.z], [os.z]; os.mcp_opts...)
        end

        surfacechange_residual!(os, sol.zero[1], delta_V)

        # Residual must be less than 10 μm sea level in piecewise linear approximation.
        # Otherwise, use piecewise constant approximation = very rare exception.
        if os.residual < 1e-5 * A_OCEAN_PD
            os.z = sol.zero[1]
            os.A = os.ref.A_itp(os.z)
        else
            os.z += delta_V / os.A
            os.A = os.ref.A_itp(os.z)
        end
    end

    return nothing
end

"""
    surfacechange_residual!(os, z, delta_V)
    surfacechange_residual!(Vresidual, z_sol, z_cur, A_itp, delta_V)

Update the residual of the piecewise linear approximation, used to solve the
sea-level/ocean-surface nonlinearity.
"""
function surfacechange_residual!(os::OceanSurfaceChange, z, delta_V)
    surfacechange_residual!(os.residual, z, os.z, os.A_itp, delta_V)
    return nothing
end

function surfacechange_residual!(Vresidual, z_sol, z_cur, A_itp, delta_V)
    Vresidual[1] = (z_sol[1] - z_cur) * mean([A_itp(z_sol[1]), A_itp(z_cur)]) - delta_V
end