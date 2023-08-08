"""
    update_geoid!(fi::FastIso)

Update the geoid by convoluting the Green's function with the load anom.
"""
function update_geoid!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    fi.geostate.geoid .= samesize_conv(fi.tools.geoidgreen,
        mass_anom(fi), fi.Omega, no_mean_bc)
    return nothing
end

"""
    columnanom_ice(fi)

Compute the density-scaled anomaly of the ice column w.r.t. the reference state.
"""
function columnanom_ice(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return fi.c.rho_ice .* (fi.geostate.H_ice - fi.refgeostate.H_ice)
end

"""
    columnanom_water(fi)

Compute the density-scaled anomaly of the (liquid) water column w.r.t. the reference state.
"""
function columnanom_water(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return fi.c.rho_seawater .* (fi.geostate.H_water - fi.refgeostate.H_water)
end

"""
    columnanom_mantle(fi)

Compute the density-scaled anomaly of the mantle column w.r.t. the reference state.
"""
function columnanom_mantle(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return fi.c.rho_uppermantle .* (fi.geostate.u - fi.refgeostate.u)
end

"""
    columnanom_litho(fi)

Compute the density-scaled anomaly of the lithosphere column w.r.t. the reference state.
"""
function columnanom_litho(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return fi.c.rho_litho .* (fi.geostate.ue - fi.refgeostate.ue)
end

"""
    columnanom_load(fi)

Compute the density-scaled anomaly of the load (ice + liquid water) column w.r.t.
the reference state.
"""
function columnanom_load(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    return columnanom_ice(fi) + columnanom_water(fi) # + columnanom_sediment(fi)
end

"""
    columnanom_full(fi)

Compute the density-scaled anomaly of the all the columns (ice + liquid water + mantle)
w.r.t. the reference state.

Correction of the surface distortion is not needed here since rho * A * z / A = rho * z.
"""
function columnanom_full(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    # columnanom_mantle() depends on sign of u, which is negative for depression.
    # Therefore, we only need to add the terms below.
    return columnanom_load(fi) + columnanom_mantle(fi) +
        columnanom_litho(fi)
end

function mass_anom(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    surface = (fi.Omega.dx * fi.Omega.dy)
    return correct_surfacedisctortion(surface .* columnanom_full(fi), fi)
end

function correct_surfacedisctortion(column::M, fi::FastIso{T, M}
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    if fi.Omega.projection_correction
        return column .* fi.Omega.K .^ 2
    else
        return column
    end
end

"""
    get_geoidgreen(fi::FastIso)

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
    update_loadcolumns!(fi::FastIso, u::AbstractMatrix{T}, H_ice::AbstractMatrix{T})

Update the load columns of a `::GeoState`.
"""
function update_loadcolumns!(fi::FastIso{T, M}, H_ice::M
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    fi.geostate.H_ice .= H_ice
    if fi.interactive_geostate
        fi.geostate.H_water .= max.(fi.geostate.sealevel -
            (fi.geostate.b + H_ice), 0)
    end
    # update sediment thickness
    return nothing
end

function update_bedrock!(fi::FastIso{T, M}, u::M
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    fi.geostate.u .= u
    fi.geostate.b .= fi.refgeostate.b .+ fi.geostate.ue .+ u
    return nothing
end

"""
    update_sealevel!(fi::FastIso)

Update the sea-level by adding the various contributions.

# Reference

Coulon et al. (2021), Figure 1.
"""
function update_sealevel!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    update_slc!(fi)
    fi.geostate.sealevel = fi.refgeostate.sealevel .+ 
        fi.geostate.geoid .+ fi.geostate.slc .+
        fi.refgeostate.conservation_term
    return nothing
end

"""
    update_slc!(fi::FastIso)

Update the sea-level contribution of melting above floatation, density correction
and potential ocean volume as in [^Goelzer2020], eq. (12).
"""
function update_slc!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    update_V_af!(fi)
    update_V_pov!(fi)
    update_V_den!(fi)
    update_slc_pov!(fi)
    update_slc_den!(fi)
    fi.geostate.slc = fi.geostate.slc_af + fi.geostate.slc_pov +
        fi.geostate.slc_den
    return nothing
end

"""
    update_V_af!(fi::FastIso)

Update the ice volume above floatation as in  [^Goelzer2020], eq. (13).
Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    fi.geostate.V_af = sum( fi.geostate.H_ice .+
        min.(fi.geostate.b .- fi.refgeostate.z0, 0) .*
        (fi.c.rho_seawater / fi.c.rho_ice) .*
        (fi.Omega.dx * fi.Omega.dy) ./ (fi.Omega.K .^ 2) )
    return nothing
end

"""
    update_slc_af!(fi::FastIso)

Update the sea-level contribution of ice above floatation as in [^Goelzer2020], eq. (2).
"""
function update_slc_af!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    fi.geostate.sle_af = fi.geostate.V_af / fi.c.A_ocean *
        fi.c.rho_ice / fi.c.rho_seawater
    fi.geostate.slc_af = -( fi.geostate.sle_af - fi.refgeostate.sle_af )
    return nothing
end

"""
    update_V_pov!(fi::FastIso)

Update the potential ocean volume as in [^Goelzer2020], eq. (14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    fi.geostate.V_pov = sum( max.(fi.refgeostate.z0 .- 
        fi.geostate.b, 0) .* (fi.Omega.dx * fi.Omega.dy) ./
        (fi.Omega.K .^ 2) )
    return nothing
end

"""
    update_slc_pov!(fi::FastIso)

Update the sea-level contribution associated with the potential ocean volume as
in [^Goelzer2020], eq. (9).
"""
function update_slc_pov!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    current = fi.geostate.V_pov / fi.c.A_ocean
    reference = fi.refgeostate.V_pov / fi.c.A_ocean
    fi.geostate.slc_pov = -(current - reference)
    return nothing
end

"""
    update_V_den!(fi::FastIso)

Update the ocean volume associated with the density correction as in [^Goelzer2020], eq. (10).
"""
function update_V_den!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    density_factor = fi.c.rho_ice / fi.c.rho_water -
        fi.c.rho_ice / fi.c.rho_seawater
    fi.geostate.V_den = sum( fi.geostate.H_ice .* density_factor ./
        (fi.Omega.K .^ 2) .* (fi.Omega.dx * fi.Omega.dy) )
    return nothing
end

"""
    update_slc_den!(fi::FastIso)

Update the sea-level contribution associated with the density correction as
in [^Goelzer2020], eq. (11).
"""
function update_slc_den!(fi::FastIso{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    current = fi.geostate.V_den / fi.c.A_ocean
    reference = fi.refgeostate.V_den / fi.c.A_ocean
    fi.geostate.slc_den = -( current - reference )
    return nothing
end