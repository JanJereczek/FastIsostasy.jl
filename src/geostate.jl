"""

    update_geostate!(gs::GeoState, u::Matrix, H_ice::Matrix, Omega::ComputationDomain,
        c::PhysicalConstants, p::MultilayerEarth, tools::PrecomputedFastiso)

Update the `::GeoState` computing the current geoid perturbation, the sea-level changes
and the load columns for the next time step of the isostasy integration.
"""
function update_geostate!(
    gs::GeoState{T},
    u::Matrix{T},
    H_ice::Matrix{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    tools::PrecomputedFastiso{T},
) where {T<:AbstractFloat}
    update_geoid!(gs, Omega, c, p, tools)
    update_sealevel!(gs, Omega, c)
    update_loadcolumns!(gs, u, H_ice)
    return nothing
end

"""

    update_geoid!(gs::GeoState, params::ODEParams)

Update the geoid of a `::GeoState` by convoluting the Green's function with the load change.
"""
function update_geoid!(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    tools::PrecomputedFastiso{T},
) where {T<:AbstractFloat}
    gs.geoid .= conv(
        tools.geoidgreen,
        get_load_change(gs, Omega, c, p),
    )[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
    return nothing
end

"""

    get_load_change(gs::GeoState, Omega::ComputationDomain, c::PhysicalConstants,
        p::MultilayerEarth)

Compute the load change compared to the reference configuration.

# Reference

Coulon et al. 2021.
"""
function get_load_change(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
) where {T<:AbstractFloat}
    return (Omega.dx * Omega.dy) .* (
        c.ice_density .* (gs.H_ice - gs.H_ice_ref) + 
        c.seawater_density .* (gs.H_water - gs.H_water_ref) +
        p.mean_density .* (gs.b - gs.b_ref)
    )
end

# TODO: transform results with stereographic utils for test2!
"""

    get_geoidgreen(Omega::ComputationDomain, c::PhysicalConstants)

Return the Green's function used to compute the changes in geoid.

# Reference

Coulon et al. 2021.
"""
function get_geoidgreen(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    geoidgreen = get_geoidgreen(R, c)
    max_geoidgreen = get_geoidgreen(mean([Omega.dx, Omega.dy]), c)  # tolerance = resolution
    return min.(geoidgreen, max_geoidgreen)
    # equivalent to: geoidgreen[geoidgreen .> max_geoidgreen] .= max_geoidgreen
end

function get_geoidgreen(R, c::PhysicalConstants{T}) where {T<:AbstractFloat}
    return c.r_equator ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_equator) ) )
end

"""

    update_loadcolumns!(gs::GeoState, u::AbstractMatrix, H_ice::AbstractMatrix)

Update the load columns of a `::GeoState`.
"""
function update_loadcolumns!(
    gs::GeoState{T},
    u::Matrix{T},
    H_ice::Matrix{T},
) where {T<:AbstractFloat}
    gs.b .= gs.b_ref .+ u
    gs.H_ice .= H_ice
    gs.H_water .= gs.sealevel - gs.b
end

"""

    update_sealevel!(gs::GeoState)

Update the sea-level `::GeoState` by adding the various contributions.

# Reference

Coulon et al. (2021), Figure 1.
"""
function update_sealevel!(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    update_slc!(gs, Omega, c)
    gs.sealevel = gs.sealevel_ref + gs.geoid + gs.slc + gs.conservation_term
    return nothing
end

"""

    update_slc!(gs::GeoState, Omega::ComputationDomain, c::PhysicalConstants)

Update the sea-level contribution of melting above floatation, density correction
and potential ocean volume.

# Reference

Goelzer et al. (2020), eq. (12).
"""
function update_slc!(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    update_V_af!(gs, Omega, c)
    update_V_pov!(gs, Omega)
    update_V_den!(gs, Omega, c)
    update_slc_pov!(gs, c)
    update_slc_den!(gs, c)
    gs.slc = gs.slc_af + gs.slc_pov + gs.slc_den
    return nothing
end

"""

    update_V_af!(gs::GeoState, Omega::ComputationDomain, c::PhysicalConstants)

Update the ice volume above floatation.

# Reference

Goelzer et al. (2020), eq. (13).
Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    gs.V_af = sum( gs.H_ice .+ min.(gs.b .- gs.z0, T(0.0)) .*
        (c.seawater_density / c.ice_density) .* (Omega.dx * Omega.dy) ./ (Omega.K .^ 2) )
    return nothing
end

"""

    update_slc_af!(gs::GeoState, c::PhysicalConstants)

Update the sea-level contribution of ice above floatation.

# Reference

Goelzer et al. (2020), eq. (2).
"""
function update_slc_af!(
    gs::GeoState{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    gs.sle_af = gs.V_af / c.A_ocean * c.ice_density / c.seawater_density
    gs.slc_af = -( gs.sle_af - gs.sle_af_ref )
    return nothing
end

"""

    update_V_pov!(gs::GeoState, Omega::ComputationDomain)

Update the potential ocean volume.

# Reference

Goelzer et al. (2020), eq. (14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
) where {T<:AbstractFloat}
    gs.V_pov = sum( max.(gs.z0 .- gs.b, T(0.0)) .*
        (Omega.dx*Omega.dy)./(Omega.K .^ 2) )
    return nothing
end

"""

    update_slc_pov!(gs::GeoState, c::PhysicalConstants)

Update the sea-level contribution associated with the potential ocean volume.

# Reference

Goelzer et al. (2020), eq. (9).
"""
function update_slc_pov!(
    gs::GeoState{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    current = gs.V_pov / c.A_ocean
    reference = gs.V_pov_ref / c.A_ocean
    gs.slc_pov = -(current - reference)
    return nothing
end

"""

    update_V_den!(gs::GeoState, Omega::ComputationDomain, c::PhysicalConstants)

Update the ocean volume associated with the density correction.

# Reference

Goelzer et al. (2020), eq. (10).
"""
function update_V_den!(
    gs::GeoState{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    density_factor = c.ice_density / c.water_density - c.ice_density / c.seawater_density
    gs.V_den = sum( gs.H_ice .* density_factor ./ (Omega.K .^ 2) .* (Omega.dx * Omega.dy) )
    return nothing
end

"""

    update_slc_den!(gs::GeoState, Omega::ComputationDomain, c::PhysicalConstants)

Update the sea-level contribution associated with the density correction.

# Reference

Goelzer et al. (2020), eq. (11).
"""
function update_slc_den!(
    gs::GeoState{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}
    current = gs.V_den / c.A_ocean
    reference = gs.V_den_ref / c.A_ocean
    gs.slc_den = -( current - reference )
    return nothing
end

function ReferenceGeoState(gs::GeoState{T}) where {T<:AbstractFloat}
    return ReferenceGeoState(
        gs.H_ice, gs.H_water, gs.b,
        gs.z0, gs.sealevel, gs.V_af, gs.sle_af,
        gs.V_pov, gs.V_den, gs.conservation_term,
    )
end