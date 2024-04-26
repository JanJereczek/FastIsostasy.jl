"""
    update_geoid!(fip::FastIsoProblem)

Update the geoid by convoluting the Green's function with the load anom.
"""
function update_geoid!(fip::FastIsoProblem)
    fip.now.geoid .= samesize_conv(mass_anom(fip), fip.tools.geoidconvo, fip.Omega)
    return nothing
end

"""
    get_geoidgreen(fip::FastIsoProblem)

Return the Green's function used to compute the geoid anomaly as in [^Coulon2021].
"""
function get_geoidgreen(Omega::ComputationDomain{T, L, M}, c::PhysicalConstants{T}
    ) where {T<:AbstractFloat, L, M<:KernelMatrix{T}}
    geoidgreen = unbounded_geoidgreen(Omega.R, c)
    max_geoidgreen = unbounded_geoidgreen(norm([100e3, 100e3]), c)  # tolerance = resolution on 100km
    return min.(geoidgreen, max_geoidgreen)
    # equivalent to: geoidgreen[geoidgreen .> max_geoidgreen] .= max_geoidgreen
end

function unbounded_geoidgreen(R, c::PhysicalConstants{<:AbstractFloat})
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

# Functions to compute/update density-scaled column anomalies
# Correction of surface distortion not needed here since rho * A * z / A = rho * z.
function anom!(x, scale, now, ref)
    @. x = scale * (now - ref)
    return nothing
end

function columnanom_mantle!(fip::FastIsoProblem)
    anom!(fip.now.columnanoms.mantle, fip.c.rho_uppermantle, fip.now.u, fip.ref.u)
    return nothing
end

function columnanom_litho!(fip::FastIsoProblem)
    anom!(fip.now.columnanoms.litho, fip.c.rho_litho, fip.now.ue, fip.ref.ue)
    return nothing
end

function columnanom_load!(fip::FastIsoProblem)
    @. fip.now.columnanoms.load = fip.ref.maskactive * (fip.c.rho_ice *
        (fip.now.H_ice - fip.ref.H_ice) + fip.c.rho_seawater * (fip.now.H_water
        - fip.ref.H_water))
    return nothing
end

function columnanom_full!(fip::FastIsoProblem)
    canoms = fip.now.columnanoms
    @. canoms.full = canoms.load + canoms.litho + canoms.mantle
    return nothing
end

function mass_anom(fip::FastIsoProblem)
    return fip.Omega.A .* (fip.now.columnanoms.full .-
        fip.c.rho_seawater .* fip.now.bsl .* fip.now.maskocean .* fip.ref.maskactive )
end

"""
    update_loadcolumns!(fip::FastIsoProblem, u::KernelMatrix{T}, H_ice::KernelMatrix{T})

Update the load columns of a `::CurrentState`.
"""
function update_loadcolumns!(fip::FastIsoProblem{T, L, M, C, FP, IP}, H_ice::M) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}

    fip.now.H_ice .= H_ice
    # fip.now.H_sed .= H_sed
    if fip.opts.interactive_sealevel
        update_maskgrounded!(fip)
        update_maskocean!(fip)
        fip.now.H_water .= watercolumn(fip)
    end
    return nothing
end

function watercolumn(fip)
    now = fip.now
    return watercolumn(now.H_ice, now.maskgrounded, now.b, now.seasurfaceheight, fip.c)
end

function watercolumn(H_ice, maskgrounded, b, seasurfaceheight, c)
    wcl = max.(seasurfaceheight .- b, 0)
    return (H_ice .<= 1) .* wcl + not.(maskgrounded) .* (H_ice .> 1) .* (wcl .-
        (H_ice .* c.rho_ice ./ c.rho_seawater))
end

function update_maskgrounded!(fip::FastIsoProblem)
    now = fip.now
    if fip.Omega.use_cuda
        now.maskgrounded .= get_maskgrounded(now, fip.c)
    else
        fip.now.maskgrounded .= collect(get_maskgrounded(now, fip.c))
    end
end

get_maskgrounded(state, c) = height_above_floatation(state, c) .> 0

function get_maskgrounded(H_ice, b, seasurfaceheight, c)
    return height_above_floatation(H_ice, b, seasurfaceheight, c) .> 0
end

function height_above_floatation(state::GeoState, c::PhysicalConstants)
    return height_above_floatation(state.H_ice, state.b,
        state.seasurfaceheight .+ state.bsl, c)
end

function height_above_floatation(H_ice, b, seasurfaceheight, c)
    return H_ice .+ min.(b .- seasurfaceheight, 0) .* (c.rho_seawater / c.rho_ice)
end

function update_maskocean!(fip)
    now = fip.now
    if fip.Omega.use_cuda
        now.maskocean .= get_maskocean(now.seasurfaceheight, now.b, now.maskgrounded)
    else
        now.maskocean .= collect(get_maskocean(now.seasurfaceheight, now.b, now.maskgrounded))
    end
end

function get_maskocean(seasurfaceheight, b, maskgrounded)
    return ((seasurfaceheight - b) .> 0) .& not.(maskgrounded)
end

function update_bedrock!(fip::FastIsoProblem{T, L, M, C, FP, IP}, u::M) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    fip.now.u .= u
    @. fip.now.b = fip.ref.b + fip.now.ue + fip.now.u
    return nothing
end

"""
    update_seasurfaceheight!(fip::FastIsoProblem)

Update the sea-level by adding the various contributions as in [coulon-contrasting-2021](@cite).
Here, the constant term is used to impose a zero geoid perturbation in the far field rather
thank for mass conservation and is embedded in convolution operation.
"""
function update_seasurfaceheight!(fip::FastIsoProblem)
    now = fip.now
    @. now.seasurfaceheight = fip.ref.seasurfaceheight + now.geoid + now.bsl
    return nothing
end

"""
    update_bsl!(fip::FastIsoProblem)

Update the sea-level contribution of melting above floatation and density correction.
Note that this differs from [goelzer-brief-2020](@cite) (eq. 12) because the ocean
surface is not assumed to be constant. Furthermore, the contribution to ocean volume
from the bedrock uplift is not included here since the volume displaced on site
is arguably blanaced by the depression of the peripherial forebulge.
"""
function update_bsl!(fip::FastIsoProblem)
    Vold = total_volume(fip)
    update_V_af!(fip)
    update_V_pov!(fip)
    update_V_den!(fip)
    Vnew = total_volume(fip)

    delta_V = Vnew - Vold
    delta_V_ocean = -delta_V
    fip.now.osc(delta_V_ocean)

    fip.now.bsl = fip.now.osc.z_k
    return nothing
end

total_volume(fip::FastIsoProblem) = fip.now.V_af * fip.c.rho_ice / fip.c.rho_seawater +
    fip.now.V_den # + fip.now.V_pov

"""
    update_V_af!(fip::FastIsoProblem)

Update the volume contribution from ice above floatation as in [goelzer-brief-2020](@cite) (eq. 13).
Note: we do not use (eq. 1) as it is only a special case of (eq. 13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(fip::FastIsoProblem)
    fip.now.V_af = sum( 
        (height_above_floatation(fip.now, fip.c) -
        height_above_floatation(fip.ref, fip.c)) .* fip.Omega.A )
    return nothing
end

"""
    update_V_den!(fip::FastIsoProblem)

Update the volume contribution associated with the density difference between meltwater and
sea water, as in [goelzer-brief-2020](@cite) (eq. 10).
"""
function update_V_den!(fip::FastIsoProblem)
    density_factor = fip.c.rho_ice / fip.c.rho_water - fip.c.rho_ice / fip.c.rho_seawater
    fip.now.V_den = sum( (fip.now.H_ice .- fip.ref.H_ice) .* density_factor .* fip.Omega.A )
    return nothing
end

"""
    update_V_pov!(fip::FastIsoProblem)

Update the volume contribution to the ocean (from isostatic adjustement in ocean regions),
which corresponds to the "potential ocean volume" in [goelzer-brief-2020](@cite) (eq. 14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(fip::FastIsoProblem)
    fip.now.V_pov = sum( max.(fip.ref.b - fip.now.b, 0) .*
        (fip.now.b .< fip.now.seasurfaceheight) .* fip.Omega.A )
    return nothing
end