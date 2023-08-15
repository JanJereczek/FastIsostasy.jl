"""
    update_geoid!(fip::FastIsoProblem)

Update the geoid by convoluting the Green's function with the load anom.
"""
function update_geoid!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    fip.geostate.geoid .= samesize_conv(fip.tools.geoidgreen,
        mass_anom(fip), fip.Omega, no_mean_bc)
    return nothing
end

"""
    columnanom_ice(fip)

Compute the density-scaled anomaly of the ice column w.r.t. the reference state.
"""
function columnanom_ice(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    return fip.c.rho_ice .* (fip.geostate.H_ice - fip.refgeostate.H_ice)
end

"""
    columnanom_water(fip)

Compute the density-scaled anomaly of the (liquid) water column w.r.t. the reference state.
"""
function columnanom_water(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    return fip.c.rho_seawater .* (fip.geostate.H_water - fip.refgeostate.H_water)
end

"""
    columnanom_mantle(fip)

Compute the density-scaled anomaly of the mantle column w.r.t. the reference state.
"""
function columnanom_mantle(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    return fip.c.rho_uppermantle .* (fip.geostate.u - fip.refgeostate.u)
end

"""
    columnanom_litho(fip)

Compute the density-scaled anomaly of the lithosphere column w.r.t. the reference state.
"""
function columnanom_litho(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    return fip.c.rho_litho .* (fip.geostate.ue - fip.refgeostate.ue)
end

"""
    columnanom_load(fip)

Compute the density-scaled anomaly of the load (ice + liquid water) column w.r.t.
the reference state.
"""
function columnanom_load(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    return columnanom_ice(fip) + columnanom_water(fip) # + columnanom_sediment(fip)
end

"""
    columnanom_full(fip)

Compute the density-scaled anomaly of the all the columns (ice + liquid water + mantle)
w.r.t. the reference state.

Correction of the surface distortion is not needed here since rho * A * z / A = rho * z.
"""
function columnanom_full(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    # columnanom_mantle() depends on sign of u, which is negative for depression.
    # Therefore, we only need to add the terms below.
    return columnanom_load(fip) + columnanom_mantle(fip) +
        columnanom_litho(fip)
end

function mass_anom(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    surface = (fip.Omega.dx * fip.Omega.dy)
    return correct_surfacedisctortion(surface .* columnanom_full(fip), fip)
end

function correct_surfacedisctortion(column::M, fip::FastIsoProblem{T, M}
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    if fip.Omega.projection_correction
        return column .* fip.Omega.K .^ 2
    else
        return column
    end
end

"""
    get_geoidgreen(fip::FastIsoProblem)

Return the Green's function used to compute the geoid anomaly as in [^Coulon2021].
"""
function get_geoidgreen(Omega::ComputationDomain{T, M}, c::PhysicalConstants{T}
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    geoidgreen = unbounded_geoidgreen(Omega.R, c)
    max_geoidgreen = unbounded_geoidgreen(norm([Omega.dx, Omega.dy]), c)  # tolerance = resolution
    return min.(geoidgreen, max_geoidgreen)
    # equivalent to: geoidgreen[geoidgreen .> max_geoidgreen] .= max_geoidgreen
end

function unbounded_geoidgreen(R, c::PhysicalConstants{<:AbstractFloat})
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

"""
    update_loadcolumns!(fip::FastIsoProblem, u::KernelMatrix{T}, H_ice::KernelMatrix{T})

Update the load columns of a `::GeoState`.
"""
function update_loadcolumns!(fip::FastIsoProblem{T, M}, H_ice::M
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    fip.geostate.H_ice .= H_ice
    # if fip.interactive_sealevel
    #     fip.geostate.H_water .= max.(fip.geostate.sealevel -
    #         (fip.geostate.b + H_ice), 0)
    # end
    # update sediment thickness
    return nothing
end

function update_bedrock!(fip::FastIsoProblem{T, M}, u::M) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    fip.geostate.u .= u
    fip.geostate.b .= fip.refgeostate.b .+ fip.geostate.ue .+ fip.geostate.u
    return nothing
end

"""
    update_sealevel!(fip::FastIsoProblem)

Update the sea-level by adding the various contributions.

# Reference

Coulon et al. (2021), Figure 1.
"""
function update_sealevel!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    update_slc!(fip)
    fip.geostate.sealevel = fip.refgeostate.sealevel .+ 
        fip.geostate.geoid .+ fip.geostate.slc .+
        fip.refgeostate.conservation_term
    return nothing
end

"""
    update_slc!(fip::FastIsoProblem)

Update the sea-level contribution of melting above floatation, density correction
and potential ocean volume as in [^Goelzer2020], eq. (12).
"""
function update_slc!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    update_V_af!(fip)
    update_V_pov!(fip)
    update_V_den!(fip)
    update_slc_pov!(fip)
    update_slc_den!(fip)
    fip.geostate.slc = fip.geostate.slc_af + fip.geostate.slc_pov +
        fip.geostate.slc_den
    return nothing
end

"""
    update_V_af!(fip::FastIsoProblem)

Update the ice volume above floatation as in  [^Goelzer2020], eq. (13).
Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    fip.geostate.V_af = sum( fip.geostate.H_ice .+
        min.(fip.geostate.b .- fip.refgeostate.z0, 0) .*
        (fip.c.rho_seawater / fip.c.rho_ice) .*
        (fip.Omega.dx * fip.Omega.dy) ./ (fip.Omega.K .^ 2) )
    return nothing
end

"""
    update_slc_af!(fip::FastIsoProblem)

Update the sea-level contribution of ice above floatation as in [^Goelzer2020], eq. (2).
"""
function update_slc_af!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    fip.geostate.sle_af = fip.geostate.V_af / fip.c.A_ocean *
        fip.c.rho_ice / fip.c.rho_seawater
    fip.geostate.slc_af = -( fip.geostate.sle_af - fip.refgeostate.sle_af )
    return nothing
end

"""
    update_V_pov!(fip::FastIsoProblem)

Update the potential ocean volume as in [^Goelzer2020], eq. (14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    fip.geostate.V_pov = sum( max.(fip.refgeostate.z0 .- 
        fip.geostate.b, 0) .* (fip.Omega.dx * fip.Omega.dy) ./
        (fip.Omega.K .^ 2) )
    return nothing
end

"""
    update_slc_pov!(fip::FastIsoProblem)

Update the sea-level contribution associated with the potential ocean volume as
in [^Goelzer2020], eq. (9).
"""
function update_slc_pov!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    current = fip.geostate.V_pov / fip.c.A_ocean
    reference = fip.refgeostate.V_pov / fip.c.A_ocean
    fip.geostate.slc_pov = -(current - reference)
    return nothing
end

"""
    update_V_den!(fip::FastIsoProblem)

Update the ocean volume associated with the density correction as in [^Goelzer2020], eq. (10).
"""
function update_V_den!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    density_factor = fip.c.rho_ice / fip.c.rho_water -
        fip.c.rho_ice / fip.c.rho_seawater
    fip.geostate.V_den = sum( fip.geostate.H_ice .* density_factor ./
        (fip.Omega.K .^ 2) .* (fip.Omega.dx * fip.Omega.dy) )
    return nothing
end

"""
    update_slc_den!(fip::FastIsoProblem)

Update the sea-level contribution associated with the density correction as
in [^Goelzer2020], eq. (11).
"""
function update_slc_den!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    current = fip.geostate.V_den / fip.c.A_ocean
    reference = fip.refgeostate.V_den / fip.c.A_ocean
    fip.geostate.slc_den = -( current - reference )
    return nothing
end