
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
    lc::GeoState{T},
) where {T<:AbstractFloat}
    return conv(
        tools.geoidgreen,
        get_load_change(Omega, c, p, lc),
    )[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end

function get_load_change(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    gs::GeoState{T},
) where {T<:AbstractFloat}
    return (Omega.dx * Omega.dy) .* (c.ice_density .* (gs.hi - gs.hi_ref) + 
        c.seawater_density .* (gs.hw - gs.hw_ref) +
        p.mean_density .* (gs.b - gs.b_ref) )
end

# TODO: for test 2, I observe distortions compared to Spada in the far-field because I don't transform with stereographic!
function get_geoidgreen(
    theta::AbstractMatrix{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    eps = 1e-12
    return c.r_equator ./ ( 2 .* c.mE .* sin.(theta ./ 2 .+ eps) )
end

function get_geoidgreen(
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



function update_loadcolumns!(
    geostate::GeoState{T},
    u::AbstractMatrix{T},
    H_ice::AbstractMatrix{T},
) where {T<:AbstractFloat}
    geostate.b = u
    geostate.hi = H_ice
end

function update_geoid!(
    geostate::GeoState{T},
    params::ODEParams{T},
) where {T<:AbstractFloat}
    geostate.geoid .= conv(
        params.tools.geoidgreen,
        get_load_change(params.Omega, params.c, params.p, geostate),
    )[params.Omega.N2:end-params.Omega.N2, params.Omega.N2:end-params.Omega.N2]
    return nothing
end

function update_sealevel!(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    geostate::GeoState{T},
    slc_af::T,
) where {T<:AbstractFloat}
    update_volume_pov!(Omega, geostate)
    update_volume_den!(Omega, c, geostate)
    update_slc_pov!(c, geostate)
    update_slc_den!(c, geostate)
    geostate.sealevel = slc_af + geostate.slc_pov + geostate.slc_den
    return nothing
end

function update_volume_pov!(
    Omega::ComputationDomain{T},
    geostate::GeoState{T},
) where {T<:AbstractFloat}
    geostate.volume_pov = sum( 
        max.(-geostate.b, T(0.0)) ./ (Omega.kn .^ 2) .* (Omega.dx * Omega.dy) )
    return nothing
end

function update_volume_den!(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    geostate::GeoState{T},
) where {T<:AbstractFloat}
    density_factor = c.ice_density / c.water_density - c.ice_density / c.seawater_density
    geostate.volume_den = sum( 
        geostate.hi .* density_factor ./ (Omega.kn .^ 2) .* (Omega.dx * Omega.dy) )
    return nothing
end

function update_slc_pov!(
    c::PhysicalConstants{T},
    geostate::GeoState{T},
) where {T<:AbstractFloat}
    current = geostate.volume_pov / c.A_ocean
    reference = geostate.volume_pov_ref / c.A_ocean
    geostate.slc_pov = -(current - reference)
    return nothing
end

function update_slc_den!(
    c::PhysicalConstants{T},
    geostate::GeoState{T},
) where {T<:AbstractFloat}
    current = geostate.volume_den / c.A_ocean
    reference = geostate.volume_den_ref / c.A_ocean
    geostate.slc_den = -( current - reference )
    return nothing
end
