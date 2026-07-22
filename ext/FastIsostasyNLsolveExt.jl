module FastIsostasyNLsolveExt

# NLsolve.jl driver for `PiecewiseLinearOceanSurfaceBSL` (barystatic_sealevel.jl).
#
# Unlike `PiecewiseConstantBSL`, which uses the ocean area at the *current* BSL
# for the whole increment, the piecewise-linear update accounts for the area
# changing across the increment: flooding from `z_cur` to `z_new` displaces the
# volume `(z_new - z_cur) * mean(A(z_cur), A(z_new))`. Matching that to the ice
# volume change `delta_V` is a scalar nonlinear equation, solved here with a
# box-constrained MCP solve (`NLsolve.mcpsolve`). The `PiecewiseLinearOceanSurfaceBSL`
# struct lives in core; its constructor and `update_bsl!` method are only defined
# once `NLsolve` is loaded, hence this extension.

using NLsolve: mcpsolve
import FastIsostasy: update_bsl!, PiecewiseLinearOceanSurfaceBSL
using FastIsostasy: ReferenceBSL, interpolate, A_OCEAN_PD

# Default options for the mixed-complementarity solve.
const DEFAULT_MCP_OPTS = (reformulation = :smooth, autodiff = :forward,
    iterations = 100_000, ftol = 1e-5, xtol = 1e-5)

function PiecewiseLinearOceanSurfaceBSL(; ref = ReferenceBSL(),
        mcp_opts = DEFAULT_MCP_OPTS)
    T = eltype(ref)
    # `residual` starts at `typemax(T)` (no solve accepted yet); `z`/`A` at the
    # reference state.
    return PiecewiseLinearOceanSurfaceBSL(ref, T(ref.z), T(ref.A), typemax(T), mcp_opts)
end

_ocean_area(z, ref) = interpolate(z, ref.A_itp)

# Volume residual of moving the barystatic sea level from `z_cur` to `z_new`
# against a target ocean-volume change `delta_V`.
function surfacechange_residual(z_new, z_cur, ref, delta_V)
    A_mean = (_ocean_area(z_new, ref) + _ocean_area(z_cur, ref)) / 2
    return (z_new - z_cur) * A_mean - delta_V
end

function update_bsl!(bsl::PiecewiseLinearOceanSurfaceBSL, delta_V, t)
    delta_V == 0 && return nothing
    ref = bsl.ref
    resid!(V, z) = (V[1] = surfacechange_residual(z[1], bsl.z, ref, delta_V); nothing)

    # Box-constrain the root by the sign of delta_V: rising sea level searches
    # in [z_cur, z_max], falling in [z_min, z_cur].
    if delta_V > 0
        sol = mcpsolve(resid!, [bsl.z], [maximum(ref.z_vec)], [bsl.z]; bsl.mcp_opts...)
    else
        sol = mcpsolve(resid!, [minimum(ref.z_vec)], [bsl.z], [bsl.z]; bsl.mcp_opts...)
    end
    z_new = sol.zero[1]
    bsl.residual = surfacechange_residual(z_new, bsl.z, ref, delta_V)

    # Accept the nonlinear solve only if its volume residual is below 10 μm of
    # equivalent sea level; otherwise fall back to the piecewise-constant update
    # (a very rare exception).
    if abs(bsl.residual) < 1e-5 * A_OCEAN_PD
        bsl.z = z_new
    else
        bsl.z += delta_V / bsl.A
    end
    bsl.A = _ocean_area(bsl.z, ref)
    return nothing
end

end # module
