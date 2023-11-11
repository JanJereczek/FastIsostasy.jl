"""
    update_geoid!(fip::FastIsoProblem)

Update the geoid by convoluting the Green's function with the load anom.
"""
function update_geoid!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat,
    M<:KernelMatrix{T}}
    fip.geostate.geoid .= samesize_conv(fip.tools.geoidgreen,
        mass_anom(fip), fip.Omega, corner_bc)
    return nothing
end

"""
    columnanom_ice!(fip)

Update the density-scaled anomaly of the ice column w.r.t. the reference state.
"""
function columnanom_ice!(fip::FastIsoProblem)
    fip.geostate.columnanoms.ice .= fip.c.rho_ice .* (fip.geostate.H_ice -
        fip.refgeostate.H_ice)
    return nothing
end

"""
    columnanom_water!(fip)

Update the density-scaled anomaly of the (liquid) water column w.r.t. the reference state.
"""
function columnanom_water!(fip::FastIsoProblem)
    fip.geostate.columnanoms.water .= fip.c.rho_seawater .* (fip.geostate.sealevel -
        fip.refgeostate.sealevel)
    return nothing
end

"""
    columnanom_mantle!(fip)

Update the density-scaled anomaly of the mantle column w.r.t. the reference state.
"""
function columnanom_mantle!(fip::FastIsoProblem)
    fip.geostate.columnanoms.mantle .= fip.c.rho_uppermantle .* (fip.geostate.u -
        fip.refgeostate.u)
    return nothing
end

"""
    columnanom_litho!(fip)

Update the density-scaled anomaly of the lithosphere column w.r.t. the reference state.
"""
function columnanom_litho!(fip::FastIsoProblem)
    fip.geostate.columnanoms.litho .= fip.c.rho_litho .* (fip.geostate.ue -
        fip.refgeostate.ue)
    return nothing
end

"""
    columnanom_load!(fip)

Update the density-scaled anomaly of the load (ice + liquid water) column w.r.t.
the reference state.
"""
function columnanom_load!(fip::FastIsoProblem)
    columnanom_ice!(fip)
    columnanom_water!(fip)
    canoms = fip.geostate.columnanoms
    canoms.load .= canoms.ice + canoms.water # + canoms.sediment
    return nothing
end

"""
    columnanom_full!(fip)

Update the density-scaled anomaly of the all the columns (ice + liquid water + mantle)
w.r.t. the reference state.

Correction of the surface distortion is not needed here since rho * A * z / A = rho * z.
"""
function columnanom_full!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    canoms = fip.geostate.columnanoms
    canoms.full .= canoms.load + canoms.litho + canoms.mantle
    return nothing
end

function mass_anom(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    surface = (fip.Omega.dx * fip.Omega.dy)
    return surface .* fip.geostate.columnanoms.full .* fip.Omega.K .^ 2
end

"""
    get_geoidgreen(fip::FastIsoProblem)

Return the Green's function used to compute the geoid anomaly as in [^Coulon2021].
"""
function get_geoidgreen(Omega::ComputationDomain{T, M}, c::PhysicalConstants{T}
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    geoidgreen = unbounded_geoidgreen(Omega.R, c)
    max_geoidgreen = unbounded_geoidgreen(norm([100e3, 100e3]), c)  # tolerance = resolution on 100km
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
    if fip.interactive_sealevel
        update_mask_grounded!(fip)
        fip.geostate.H_water .= not.(fip.geostate.maskgrounded) .* fip.geostate.sealevel
    end
    ########
    # update sediment thickness
    ########
    return nothing
end

function update_mask_grounded!(fip::FastIsoProblem)
    fip.geostate.maskgrounded .= height_above_floatation(fip) .> 0
end

function height_above_floatation(fip::FastIsoProblem)
    return fip.geostate.H_ice .+ min.(fip.geostate.b .- fip.refgeostate.z0, 0) .*
        (fip.c.rho_seawater / fip.c.rho_ice)
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
function update_sealevel!(fip::FastIsoProblem{T, M}) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    update_slc!(fip)
    gs = fip.geostate
    gs.sealevel .= gs.sealevel .+ gs.geoid .+ gs.bsl
    return nothing
end

"""
    update_slc!(fip::FastIsoProblem)

Update the sea-level contribution of melting above floatation, density correction
and potential ocean volume as in [^Goelzer2020], eq. (12).
"""
function update_slc!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    Vold = total_volume(fip)
    update_V_af!(fip)
    update_V_pov!(fip)
    update_V_den!(fip)
    Vnew = total_volume(fip)

    delta_V = Vnew - Vold
    delta_V_ocean = -delta_V
    fip.geostate.osc(delta_V_ocean)

    fip.geostate.bsl = fip.geostate.osc.z_k
    return nothing
end

total_volume(fip::FastIsoProblem) = fip.geostate.V_af * fip.c.rho_ice /
    fip.c.rho_seawater + fip.geostate.V_den + fip.geostate.V_pov

"""
    update_V_af!(fip::FastIsoProblem)

Update the ice volume above floatation as in  [^Goelzer2020], eq. (13).
Note: we do not use eq. (1) as it is only a special case of eq. (13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    fip.geostate.V_af = sum( height_above_floatation(fip) .*
        (fip.Omega.dx * fip.Omega.dy) ./ (fip.Omega.K .^ 2) )
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
    update_V_den!(fip::FastIsoProblem)

Update the ocean volume associated with the density correction as in [^Goelzer2020], eq. (10).
"""
function update_V_den!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    density_factor = fip.c.rho_ice / fip.c.rho_water -
        fip.c.rho_ice / fip.c.rho_seawater
    fip.geostate.V_den = sum( fip.geostate.columnanoms.ice .* density_factor ./
        (fip.Omega.K .^ 2) .* (fip.Omega.dx * fip.Omega.dy) )
    return nothing
end