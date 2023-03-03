
"""

    compute_geoid_response(c, p, Omega, tools, lc)

Compute the geoid response to the load changes `lc`, with `Omega` the computation
domain, `c` the physical constants, `p` the solid-Earth parameters and `tools` the
precomputed terms to accelerate FastIsostasy.

=================
Reference:
=================

Coulon et al. 2021.
"""
function compute_geoid_response(
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    Omega::ComputationDomain{T},
    tools::PrecomputedFastiso{T},
    lc::ColumnHeights{T},
) where {T<:AbstractFloat}
    return conv(
        tools.geoid_green,
        get_load_change(Omega, c, p, lc),
    )[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end

function get_load_change(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    lc::ColumnHeights{T},
) where {T<:AbstractFloat}
    return (Omega.dx * Omega.dy) .* (c.ice_density .* (lc.hi - lc.hi0) + 
        c.seawater_density .* (lc.hw - lc.hw0) +
        p.mean_density .* (lc.b - lc.b0) )
end

# TODO: for test 2, I observe distortions compared to Spada in the far-field because I don't transform with stereographic!
function get_geoid_green(
    theta::AbstractMatrix{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    eps = 1e-12
    return c.r_equator ./ ( 2 .* c.mE .* sin.(theta ./ 2 .+ eps) )
end

function get_geoid_green(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    geoid = c.r_equator ./ ( 2 .* c.mE .* sin.( Omega.R ./ (2 .* c.r_equator) ) )

    # Set the resolution as tolerance for the computation of the geoid's Green function
    max_geoid = c.r_equator ./
        ( 2 .* c.mE .* sin.( mean([Omega.dx, Omega.dy]) ./ (2 .* c.r_equator) ) )
    geoid[geoid .> max_geoid] .= max_geoid
    return geoid
end

function init_columnchanges(
    Omega::ComputationDomain{T};
    hi::AbstractMatrix{T} = fill(T(0), Omega.N, Omega.N),
    hi0::AbstractMatrix{T} = fill(T(0), Omega.N, Omega.N),
    hw::AbstractMatrix{T} = fill(T(0), Omega.N, Omega.N),
    hw0::AbstractMatrix{T} = fill(T(0), Omega.N, Omega.N),
    b::AbstractMatrix{T} = fill(T(0), Omega.N, Omega.N),
    b0::AbstractMatrix{T} = fill(T(0), Omega.N, Omega.N),
) where {T<:AbstractFloat}
    return ColumnHeights(hi, hi0, hw, hw0, b, b0)
end

function update_columnchanges!(
    lc::ColumnHeights{T},
    u::AbstractMatrix{T},
    H_ice::AbstractMatrix{T},
) where {T<:AbstractFloat}
    lc.b = u
    lc.hi = H_ice
end