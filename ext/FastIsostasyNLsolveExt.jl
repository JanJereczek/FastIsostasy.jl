using NLsolve: mcpsolve

###########################################################################################
# PiecewiseLinearOceanSurface
###########################################################################################

"""
    PiecewiseLinearOceanSurface{T}
    PiecewiseLinearOceanSurface(; ref, mcp_opts)

A `mutable struct` that is only available if `using NLsolve` and contains:
- `ref`: a [`ReferenceBSL`](@ref).
- `z`: the current BSL.
- `A`: the current ocean surface.
- `residual`: residual of the nonlinear equation solved numerically.
- `mcp_opts`: options for the MCP solver, such as `reformulation`, `autodiff`, `iterations`, `ftol`, and `xtol`.

Note that, unlike [`ConstantOceanSurface`](@ref) and [`PiecewiseConstantOceanSurface`](@ref), this will only work if `using NLsolve`.
"""
mutable struct PiecewiseLinearOceanSurface{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    A::T
    residual::T
    mcp_opts::NamedTuple
end

function PiecewiseLinearOceanSurface(; ref = ReferenceBSL(),
    mcp_opts = (reformulation = :smooth, autodiff = :forward,
        iterations = 100_000, ftol = 1e-5, xtol = 1e-5) )
    return PiecewiseLinearOceanSurface(ref, z, A, typemax(eltype(ref)), mcp_opts)
end

function update_ocean!(bsl::PiecewiseLinearOceanSurface, delta_V)
    scr!(Vresidual, z) = surfacechange_residual!(Vresidual, z, bsl.z, bsl.ref.A_itp, delta_V)

    # Update ocean surface within reasonable bounds defined by z_max_update and
    # only if sea-level contribution is nonzero.
    if delta_V != 0
        if delta_V > 0
            sol = mcpsolve(scr!, [bsl.z], [maximum(bsl.ref.z_vec)], [bsl.z]; bsl.mcp_opts...)
        elseif delta_V < 0
            sol = mcpsolve(scr!, [minimum(bsl.ref.z_vec)], [bsl.z], [bsl.z]; bsl.mcp_opts...)
        end

        surfacechange_residual!(bsl, sol.zero[1], delta_V)

        # Residual must be less than 10 μm sea level in piecewise linear approximation.
        # Otherwise, use piecewise constant approximation = very rare exception.
        if bsl.residual < 1e-5 * A_OCEAN_PD
            bsl.z = sol.zero[1]
            bsl.A = bsl.ref.A_itp(bsl.z)
        else
            bsl.z += delta_V / bsl.A
            bsl.A = bsl.ref.A_itp(bsl.z)
        end
    end

    return nothing
end

"""
    surfacechange_residual!(bsl, z, delta_V)
    surfacechange_residual!(Vresidual, z_sol, z_cur, A_itp, delta_V)

Update the residual of the piecewise linear approximation, used to solve the
sea-level/ocean-surface nonlinearity.
"""
function surfacechange_residual!(bsl::OceanSurfaceChange, z, delta_V)
    surfacechange_residual!(bsl.residual, z, bsl.z, bsl.A_itp, delta_V)
    return nothing
end

function surfacechange_residual!(Vresidual, z_sol, z_cur, A_itp, delta_V)
    Vresidual[1] = (z_sol[1] - z_cur) * mean([A_itp(z_sol[1]), A_itp(z_cur)]) - delta_V
end