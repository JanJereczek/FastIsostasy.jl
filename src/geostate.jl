"""

    update_geostate!(sstruct::SuperStruct, u::Matrix, H_ice::Matrix)

Update the `::GeoState` computing the current geoid perturbation, the sea-level changes
and the load columns for the next time step of the isostasy integration.
"""
function update_geostate!(
    sstruct::SuperStruct{T},
    u::XMatrix,
    H_ice::XMatrix,
) where {T<:AbstractFloat}

    update_geoid!(sstruct)
    update_sealevel!(sstruct)
    update_loadcolumns!(sstruct, u, H_ice)
    return nothing
end

"""

    update_geoid!(sstruct::SuperStruct)

Update the geoid of a `::GeoState` by convoluting the Green's function with the load change.
"""
function update_geoid!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    sstruct.geostate.geoid .= view(
        conv( sstruct.tools.geoidgreen, get_greenloadchange(sstruct) ),
        sstruct.Omega.N2:2*sstruct.Omega.N-1-sstruct.Omega.N2,
        sstruct.Omega.N2:2*sstruct.Omega.N-1-sstruct.Omega.N2,
    )
    # spada_calibration = 1.0
    # sstruct.geostate.geoid .-= (spada_calibration * maximum(sstruct.geostate.geoid))
    return nothing
end

"""

    get_loadchange(sstruct::SuperStruct)

Compute the load change compared to the reference configuration.

# Reference

Coulon et al. 2021.
Difference to Coulon: HERE WE SCALE WITH CORRECTION FACTOR FOR PROJECTION!
"""
function get_greenloadchange(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    return (sstruct.Omega.dx * sstruct.Omega.dy) .* get_fullcolumnchange(sstruct)
end

function get_fullcolumnchange(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    return get_loadcolumnchange(sstruct) + sstruct.p.mean_density[1] .*
            (sstruct.geostate.b - sstruct.refgeostate.b) # .* Omega.K

end

function get_loadcolumnchange(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    return (sstruct.c.rho_ice .* (sstruct.geostate.H_ice - sstruct.refgeostate.H_ice) +
        sstruct.c.rho_seawater .* (sstruct.geostate.H_water - sstruct.refgeostate.H_water) ) # .* Omega.K
end

function get_loadchange(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    return - sstruct.c.g .* get_loadcolumnchange(sstruct)
end

# TODO: transform results with stereographic utils for test2!
"""

    get_geoidgreen(sstruct::SuperStruct)

Return the Green's function used to compute the changes in geoid.

# Reference

Coulon et al. 2021.
"""
function get_geoidgreen(Omega::ComputationDomain, c::PhysicalConstants{T}) where {T<:AbstractFloat}
    geoidgreen = unbounded_geoidgreen(Omega.R, c)
    max_geoidgreen = unbounded_geoidgreen(norm([Omega.dx, Omega.dy]), c)  # tolerance = resolution
    return min.(geoidgreen, max_geoidgreen)
    # equivalent to: geoidgreen[geoidgreen .> max_geoidgreen] .= max_geoidgreen
end

function unbounded_geoidgreen(R, c::PhysicalConstants{T}) where {T<:AbstractFloat}
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

"""

    update_loadcolumns!(sstruct::SuperStruct, u::XMatrix, H_ice::XMatrix)

Update the load columns of a `::GeoState`.
"""
function update_loadcolumns!(
    sstruct::SuperStruct{T},
    u::XMatrix,
    H_ice::XMatrix,
) where {T<:AbstractFloat}

    sstruct.geostate.b .= sstruct.refgeostate.b .+ u
    sstruct.geostate.H_ice .= H_ice
    sstruct.geostate.H_water .= max.(sstruct.geostate.sealevel - (sstruct.geostate.b + H_ice), 0)
    return nothing
end

"""

    update_sealevel!(sstruct::SuperStruct)

Update the sea-level `::GeoState` by adding the various contributions.

# Reference

Coulon et al. (2021), Figure 1.
"""
function update_sealevel!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    update_slc!(sstruct)
    sstruct.geostate.sealevel = sstruct.refgeostate.sealevel .+ 
        sstruct.geostate.geoid .+ sstruct.geostate.slc .+
        sstruct.refgeostate.conservation_term
    return nothing
end

"""

    update_slc!(sstruct::SuperStruct)

Update the sea-level contribution of melting above floatation, density correction
and potential ocean volume.

# Reference

Goelzer et al. (2020), eq. (12).
"""
function update_slc!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    update_V_af!(sstruct)
    update_V_pov!(sstruct)
    update_V_den!(sstruct)
    update_slc_pov!(sstruct)
    update_slc_den!(sstruct)
    sstruct.geostate.slc = sstruct.geostate.slc_af + sstruct.geostate.slc_pov +
        sstruct.geostate.slc_den
    return nothing
end

"""

    update_V_af!(sstruct::SuperStruct)

Update the ice volume above floatation.

# Reference

Goelzer et al. (2020), eq. (13).
Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    sstruct.geostate.V_af = sum( sstruct.geostate.H_ice .+
        min.(sstruct.geostate.b .- sstruct.refgeostate.z0, 0) .*
        (sstruct.c.rho_seawater / sstruct.c.rho_ice) .*
        (sstruct.Omega.dx * sstruct.Omega.dy) ./ (sstruct.Omega.K .^ 2) )
    return nothing
end

"""

    update_slc_af!(sstruct::SuperStruct)

Update the sea-level contribution of ice above floatation.

# Reference

Goelzer et al. (2020), eq. (2).
"""
function update_slc_af!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    sstruct.geostate.sle_af = sstruct.geostate.V_af / sstruct.c.A_ocean *
        sstruct.c.rho_ice / sstruct.c.rho_seawater
    sstruct.geostate.slc_af = -( sstruct.geostate.sle_af - sstruct.refgeostate.sle_af )
    return nothing
end

"""

    update_V_pov!(sstruct::SuperStruct)

Update the potential ocean volume.

# Reference

Goelzer et al. (2020), eq. (14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcinsstruct.geostate.
"""
function update_V_pov!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    sstruct.geostate.V_pov = sum( max.(sstruct.refgeostate.z0 .- 
        sstruct.geostate.b, T(0.0)) .* (sstruct.Omega.dx * sstruct.Omega.dy) ./
        (sstruct.Omega.K .^ 2) )
    return nothing
end

"""

    update_slc_pov!(sstruct::SuperStruct)

Update the sea-level contribution associated with the potential ocean volume.

# Reference

Goelzer et al. (2020), eq. (9).
"""
function update_slc_pov!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    current = sstruct.geostate.V_pov / sstruct.c.A_ocean
    reference = sstruct.refgeostate.V_pov / sstruct.c.A_ocean
    sstruct.geostate.slc_pov = -(current - reference)
    return nothing
end

"""

    update_V_den!(sstruct::SuperStruct)

Update the ocean volume associated with the density correction.

# Reference

Goelzer et al. (2020), eq. (10).
"""
function update_V_den!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    density_factor = sstruct.c.rho_ice / sstruct.c.rho_water -
        sstruct.c.rho_ice / sstruct.c.rho_seawater
    sstruct.geostate.V_den = sum( sstruct.geostate.H_ice .* density_factor ./
        (sstruct.Omega.K .^ 2) .* (sstruct.Omega.dx * sstruct.Omega.dy) )
    return nothing
end

"""

    update_slc_den!(sstruct::SuperStruct)

Update the sea-level contribution associated with the density correction.

# Reference

Goelzer et al. (2020), eq. (11).
"""
function update_slc_den!(sstruct::SuperStruct{T}) where {T<:AbstractFloat}
    current = sstruct.geostate.V_den / sstruct.c.A_ocean
    reference = sstruct.refgeostate.V_den / sstruct.c.A_ocean
    sstruct.geostate.slc_den = -( current - reference )
    return nothing
end

# function ReferenceGeoState(gs::GeoState{T}) where {T<:AbstractFloat}
#     return ReferenceGeoState(
#         gs.H_ice, gs.H_water, gs.b,
#         gs.z0, gs.sealevel, gs.V_af, gs.sle_af,
#         gs.V_pov, gs.V_den, gs.conservation_term,
#     )
# end