"""
    update_dz_ss!(fip::FastIsoProblem)

Update the SSH perturbation `dz_ss` by convoluting the Green's function with the load anom.
"""
function update_dz_ss!(fip::FastIsoProblem, sl::LaterallyVariableSeaSurface)

    @. fip.tools.prealloc.buffer_x = mass_anom(fip.Omega.A, fip.now.columnanoms.full)
    samesize_conv!(fip.now.dz_ss, fip.tools.prealloc.buffer_x,
        fip.tools.dz_ss_convo, fip.Omega, fip.bcs.sea_surface_perturbation,
        fip.bcs.sea_surface_perturbation.space)
    return nothing
end

function update_dz_ss!(fip::FastIsoProblem, sl::LaterallyConstantSeaSurface)
    return nothing
end

"""
    get_dz_ss_green(fip::FastIsoProblem)

Return the Green's function used to compute the SSH perturbation `dz_ss` as in [^Coulon2021].
"""
function get_dz_ss_green(Omega::RegionalComputationDomain, c::PhysicalConstants)
    dz_ssgreen = unbounded_dz_ssgreen(Omega.R, c)
    max_dz_ssgreen = unbounded_dz_ssgreen(norm([100e3, 100e3]), c)  # tolerance = resolution on 100km
    return min.(dz_ssgreen, max_dz_ssgreen)
    # equivalent to: dz_ssgreen[dz_ssgreen .> max_dz_ssgreen] .= max_dz_ssgreen
end

function unbounded_dz_ssgreen(R, c::PhysicalConstants)
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

# Functions to compute/update density-scaled column anomalies
# Correction of surface distortion not needed here since rho * A * z / A = rho * z.
function anom!(x, scale, now, ref)
    @. x = scale * (now - ref)
    return nothing
end

function columnanom_mantle!(fip::FastIsoProblem)
    anom!(fip.now.columnanoms.mantle, fip.p.rho_uppermantle, fip.now.u, fip.ref.u)
    return nothing
end

function columnanom_litho!(fip::FastIsoProblem)
    anom!(fip.now.columnanoms.litho, fip.p.rho_litho, fip.now.ue, fip.ref.ue)
    return nothing
end

function columnanom_ice!(fip::FastIsoProblem)
    anom!(fip.now.columnanoms.ice, fip.c.rho_ice, fip.now.H_ice, fip.ref.H_ice)
    return nothing
end

function columnanom_water!(fip::FastIsoProblem, ol::InteractiveOceanLoad)
    watercolumn!(fip)
    anom!(fip.now.columnanoms.seawater, fip.c.rho_seawater, fip.now.H_water, fip.ref.H_water)
    return nothing
end

function columnanom_water!(fip::FastIsoProblem, ol::NoOceanLoad)
    watercolumn!(fip)
    fip.now.columnanoms.seawater .= 0
    return nothing
end

function watercolumn!(fip::FastIsoProblem)
    watercolumn!(fip.now.H_water, fip.now.H_ice, fip.now.maskgrounded, fip.now.z_b,
        fip.now.z_ss, fip.c, fip.tools.prealloc.buffer_x)
    return nothing
end

function watercolumn!(H_water, H_ice, maskgrounded, z_b, z_ss, c, buffer)
    # water column height in absence of ice
    buffer .= max.(z_ss .- z_b, 0)

    # if ice thickness lesser than threshold, only impose water column
    # if ice thickness greater than threshold, impose difference (accounting for floatation)
    H_water .= (H_ice .<= 1) .* buffer .+
        not.(maskgrounded) .* (H_ice .> 1) .*
        (buffer .- (H_ice .* (c.rho_ice / c.rho_seawater)))
    return nothing
end

function watercolumn(H_ice, maskgrounded, z_b, z_ss, c)
    H_water, buffer = similar(H_ice), similar(H_ice)
    watercolumn!(H_water, H_ice, maskgrounded, z_b, z_ss, c, buffer)
    return H_water
end

function columnanom_sediment!(fip::FastIsoProblem)
end

function columnanom_load!(fip::FastIsoProblem)
    canoms = fip.now.columnanoms
    @. canoms.load .= fip.ref.maskactive * (canoms.ice + canoms.seawater + canoms.sediment)
    return nothing
end

function columnanom_full!(fip::FastIsoProblem)
    canoms = fip.now.columnanoms
    @. canoms.full = canoms.load + canoms.litho + canoms.mantle
    return nothing
end

function mass_anom(fip::FastIsoProblem)
    return fip.Omega.A .* (fip.now.columnanoms.full) # .-
        # fip.c.rho_seawater .* fip.now.bsl .* fip.now.maskocean .* fip.ref.maskactive
end

function mass_anom(A, canom_full)
    return A * canom_full
end

function update_maskgrounded!(fip::FastIsoProblem)
    fip.now.maskgrounded .= fip.now.H_af .> 0
    return nothing
end

get_maskgrounded(state, c) = height_above_floatation(state, c) .> 0

function get_maskgrounded(H_ice, b, z_ss, c)
    return height_above_floatation(H_ice, b, z_ss, c) .> 0
end

function get_maskocean(z_ss, b, maskgrounded)
    return ((z_ss - b) .> 0) .& not.(maskgrounded)
end

function height_above_floatation(state::AbstractState, c::PhysicalConstants)
    return height_above_floatation(state.H_ice, state.z_b,
        state.z_ss, c)
end

function height_above_floatation(H_ice, b, z_ss, c)
    return max.(H_ice .+ min.(b .- z_ss, 0), 0) .* (c.rho_seawater / c.rho_ice)
end

function update_maskocean!(fip)
    @. fip.now.maskocean = (fip.now.z_ss - fip.now.z_b) > 0
    @. fip.now.maskocean = fip.now.maskocean .& not.(fip.now.maskgrounded)
end

function update_bedrock!(fip::FastIsoProblem, u)
    fip.now.u .= u
    @. fip.now.z_b = fip.ref.z_b + fip.now.ue + fip.now.u
    return nothing
end

function update_Haf!(fip::FastIsoProblem)
    @. fip.now.H_af = max(fip.now.H_ice + min(fip.now.z_b - fip.now.z_ss, 0), 0) 
    fip.now.H_af .*= fip.c.rho_sw_ice
    return nothing
end

"""
    update_z_ss!(fip::FastIsoProblem)

Update the sea-level by adding the various contributions as in [coulon-contrasting-2021](@cite).
Here, the constant term is used to impose a zero dz_ss perturbation in the far field rather
thank for mass conservation and is embedded in convolution operation.
"""
function update_z_ss!(fip::FastIsoProblem)
    @. fip.now.z_ss = fip.ref.z_ss + fip.now.dz_ss + fip.now.bsl.z
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
    update_bsl!(fip.now.bsl, -delta_V, fip.nout.t_steps_ode[end])
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
    fip.tools.prealloc.buffer_x .= (fip.now.H_af .- fip.ref.H_af) .* fip.Omega.A 
    fip.now.V_af = sum(fip.tools.prealloc.buffer_x)
    return nothing
end

"""
    update_V_den!(fip::FastIsoProblem)

Update the volume contribution associated with the density difference between meltwater and
sea water, as in [goelzer-brief-2020](@cite) (eq. 10).
"""
function update_V_den!(fip::FastIsoProblem)
    density_factor = fip.c.rho_ice / fip.c.rho_water - fip.c.rho_ice / fip.c.rho_seawater
    fip.tools.prealloc.buffer_x .= (fip.now.H_water .- fip.ref.H_water) .* fip.Omega.A
    fip.now.V_den = sum( fip.tools.prealloc.buffer_x ) * density_factor
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
    fip.tools.prealloc.buffer_x .= fip.now.z_b .- fip.now.z_ss
    fip.tools.prealloc.buffer_x .= max.(fip.tools.prealloc.buffer_x, 0) .* fip.Omega.A
    fip.tools.prealloc.buffer_x .= fip.tools.prealloc.buffer_x .* (fip.now.z_b .< fip.now.z_ss)
    fip.now.V_pov = sum( fip.tools.prealloc.buffer_x )
    return nothing
end