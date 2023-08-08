"""
    update_geoid!(sstruct::SuperStruct)

Update the geoid by convoluting the Green's function with the load anom.
"""
function update_geoid!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    sstruct.geostate.geoid .= samesize_conv(sstruct.tools.geoidgreen,
        mass_anom(sstruct), sstruct.Omega, no_mean_bc)
    return nothing
end

"""
    columnanom_ice(sstruct)

Compute the density-scaled anomaly of the ice column w.r.t. the reference state.
"""
function columnanom_ice(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return sstruct.c.rho_ice .* (sstruct.geostate.H_ice - sstruct.refgeostate.H_ice)
end

"""
    columnanom_water(sstruct)

Compute the density-scaled anomaly of the (liquid) water column w.r.t. the reference state.
"""
function columnanom_water(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return sstruct.c.rho_seawater .* (sstruct.geostate.H_water - sstruct.refgeostate.H_water)
end

"""
    columnanom_mantle(sstruct)

Compute the density-scaled anomaly of the mantle column w.r.t. the reference state.
"""
function columnanom_mantle(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return sstruct.c.rho_uppermantle .* (sstruct.geostate.u - sstruct.refgeostate.u)
end

"""
    columnanom_litho(sstruct)

Compute the density-scaled anomaly of the lithosphere column w.r.t. the reference state.
"""
function columnanom_litho(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return sstruct.c.rho_litho .* (sstruct.geostate.ue - sstruct.refgeostate.ue)
end

"""
    columnanom_load(sstruct)

Compute the density-scaled anomaly of the load (ice + liquid water) column w.r.t.
the reference state.
"""
function columnanom_load(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return columnanom_ice(sstruct) + columnanom_water(sstruct) # + columnanom_sediment(sstruct)
end

"""
    columnanom_full(sstruct)

Compute the density-scaled anomaly of the all the columns (ice + liquid water + mantle)
w.r.t. the reference state.

Correction of the surface distortion is not needed here since rho * A * z / A = rho * z.
"""
function columnanom_full(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    # columnanom_mantle() depends on sign of u, which is negative for depression.
    # Therefore, we only need to add the terms below.
    return columnanom_load(sstruct) + columnanom_mantle(sstruct) +
        columnanom_litho(sstruct)
end

function mass_anom(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    surface = (sstruct.Omega.dx * sstruct.Omega.dy)
    return correct_surfacedisctortion(surface .* columnanom_full(sstruct), sstruct)
end

function correct_surfacedisctortion(column::M, sstruct::SuperStruct{T, M}
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    if sstruct.Omega.projection_correction
        return column .* sstruct.Omega.K .^ 2
    else
        return column
    end
end

"""
    get_geoidgreen(sstruct::SuperStruct)

Return the Green's function used to compute the anoms in geoid.

# Reference

Coulon et al. 2021.
"""
function get_geoidgreen(Omega::ComputationDomain{T, M}, c::PhysicalConstants{T}
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    geoidgreen = unbounded_geoidgreen(Omega.R, c)
    max_geoidgreen = unbounded_geoidgreen(norm([Omega.dx, Omega.dy]), c)  # tolerance = resolution
    return min.(geoidgreen, max_geoidgreen)
    # equivalent to: geoidgreen[geoidgreen .> max_geoidgreen] .= max_geoidgreen
end

function unbounded_geoidgreen(R, c::PhysicalConstants{<:AbstractFloat})
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

"""
    update_loadcolumns!(sstruct::SuperStruct, u::AbstractMatrix{T}, H_ice::AbstractMatrix{T})

Update the load columns of a `::GeoState`.
"""
function update_loadcolumns!(sstruct::SuperStruct{T, M}, H_ice::M
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    sstruct.geostate.H_ice .= H_ice
    if sstruct.interactive_geostate
        sstruct.geostate.H_water .= max.(sstruct.geostate.sealevel -
            (sstruct.geostate.b + H_ice), 0)
    end
    # update sediment thickness
    return nothing
end

function update_bedrock!(sstruct::SuperStruct{T, M}, u::M
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    sstruct.geostate.u .= u
    sstruct.geostate.b .= sstruct.refgeostate.b .+ sstruct.geostate.ue .+ u
    return nothing
end

"""
    update_sealevel!(sstruct::SuperStruct)

Update the sea-level by adding the various contributions.

# Reference

Coulon et al. (2021), Figure 1.
"""
function update_sealevel!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    update_slc!(sstruct)
    sstruct.geostate.sealevel = sstruct.refgeostate.sealevel .+ 
        sstruct.geostate.geoid .+ sstruct.geostate.slc .+
        sstruct.refgeostate.conservation_term
    return nothing
end

"""
    update_slc!(sstruct::SuperStruct)

Update the sea-level contribution of melting above floatation, density correction
and potential ocean volume as in [^Goelzer2020], eq. (12).
"""
function update_slc!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
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

Update the ice volume above floatation as in  [^Goelzer2020], eq. (13).
Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    sstruct.geostate.V_af = sum( sstruct.geostate.H_ice .+
        min.(sstruct.geostate.b .- sstruct.refgeostate.z0, 0) .*
        (sstruct.c.rho_seawater / sstruct.c.rho_ice) .*
        (sstruct.Omega.dx * sstruct.Omega.dy) ./ (sstruct.Omega.K .^ 2) )
    return nothing
end

"""
    update_slc_af!(sstruct::SuperStruct)

Update the sea-level contribution of ice above floatation as in [^Goelzer2020], eq. (2).
"""
function update_slc_af!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    sstruct.geostate.sle_af = sstruct.geostate.V_af / sstruct.c.A_ocean *
        sstruct.c.rho_ice / sstruct.c.rho_seawater
    sstruct.geostate.slc_af = -( sstruct.geostate.sle_af - sstruct.refgeostate.sle_af )
    return nothing
end

"""
    update_V_pov!(sstruct::SuperStruct)

Update the potential ocean volume as in [^Goelzer2020], eq. (14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    sstruct.geostate.V_pov = sum( max.(sstruct.refgeostate.z0 .- 
        sstruct.geostate.b, 0) .* (sstruct.Omega.dx * sstruct.Omega.dy) ./
        (sstruct.Omega.K .^ 2) )
    return nothing
end

"""
    update_slc_pov!(sstruct::SuperStruct)

Update the sea-level contribution associated with the potential ocean volume as
in [^Goelzer2020], eq. (9).
"""
function update_slc_pov!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    current = sstruct.geostate.V_pov / sstruct.c.A_ocean
    reference = sstruct.refgeostate.V_pov / sstruct.c.A_ocean
    sstruct.geostate.slc_pov = -(current - reference)
    return nothing
end

"""
    update_V_den!(sstruct::SuperStruct)

Update the ocean volume associated with the density correction as in [^Goelzer2020], eq. (10).
"""
function update_V_den!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    density_factor = sstruct.c.rho_ice / sstruct.c.rho_water -
        sstruct.c.rho_ice / sstruct.c.rho_seawater
    sstruct.geostate.V_den = sum( sstruct.geostate.H_ice .* density_factor ./
        (sstruct.Omega.K .^ 2) .* (sstruct.Omega.dx * sstruct.Omega.dy) )
    return nothing
end

"""
    update_slc_den!(sstruct::SuperStruct)

Update the sea-level contribution associated with the density correction as
in [^Goelzer2020], eq. (11).
"""
function update_slc_den!(sstruct::SuperStruct{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    current = sstruct.geostate.V_den / sstruct.c.A_ocean
    reference = sstruct.refgeostate.V_den / sstruct.c.A_ocean
    sstruct.geostate.slc_den = -( current - reference )
    return nothing
end