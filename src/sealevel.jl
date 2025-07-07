"""
    update_dz_ss!(sim::Simulation)

Update the SSH perturbation `dz_ss` by convoluting the Green's function with the load anom.
"""
function update_dz_ss!(sim::Simulation, sl::LaterallyVariableSeaSurface)

    @. sim.tools.prealloc.buffer_x = mass_anom(sim.domain.A, sim.now.columnanoms.full)
    samesize_conv!(sim.now.dz_ss, sim.tools.prealloc.buffer_x,
        sim.tools.dz_ss_convo, sim.domain, sim.bcs.sea_surface_perturbation,
        sim.bcs.sea_surface_perturbation.space)
    return nothing
end

function update_dz_ss!(sim::Simulation, sl::LaterallyConstantSeaSurface)
    return nothing
end

"""
    get_dz_ss_green(sim::Simulation)

Return the Green's function used to compute the SSH perturbation `dz_ss` as in [^Coulon2021].
"""
function get_dz_ss_green(domain::RegionalDomain, c::PhysicalConstants)
    dz_ssgreen = unbounded_dz_ssgreen(domain.R, c)
    max_dz_ssgreen = unbounded_dz_ssgreen(norm([100e3, 100e3]), c)  # tolerance = resolution on 100km
    return min.(dz_ssgreen, max_dz_ssgreen)
    # equivalent to: dz_ssgreen[dz_ssgreen .> max_dz_ssgreen] .= max_dz_ssgreen
end

function unbounded_dz_ssgreen(R, c::PhysicalConstants)
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

"""
    update_z_ss!(sim::Simulation)

Update the sea-level by adding the various contributions as in [coulon-contrasting-2021](@cite).
Here, the constant term is used to impose a zero dz_ss perturbation in the far field rather
thank for mass conservation and is embedded in convolution operation.
"""
function update_z_ss!(sim::Simulation)
    @. sim.now.z_ss = sim.ref.z_ss + sim.now.dz_ss + sim.now.z_bsl
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the sea-level contribution of melting above floatation and density correction.
Note that this differs from [goelzer-brief-2020](@cite) (eq. 12) because the ocean
surface is not assumed to be constant. Furthermore, the contribution to ocean volume
from the bedrock uplift is not included here since the volume displaced on site
is arguably blanaced by the depression of the peripherial forebulge.
"""
function internal_update_bsl!(sim::Simulation, up::InternalUpdateBSL)
    update_delta_V!(sim)
    update_bsl!(sim.model.bsl, -sim.now.delta_V, sim.nout.t_steps_ode[end])
    sim.now.z_bsl = sim.model.bsl.z
    return nothing
end

function internal_update_bsl!(sim::Simulation, up::ExternalUpdateBSL)
    update_delta_V!(sim)
    return nothing
end

function update_delta_V!(sim::Simulation)
    Vold = total_volume(sim)
    update_V_af!(sim)
    update_V_pov!(sim)
    update_V_den!(sim)
    Vnew = total_volume(sim)
    sim.now.delta_V = Vnew - Vold
    return nothing
end

total_volume(sim::Simulation) = sim.now.V_af * sim.c.rho_ice / sim.c.rho_seawater +
    sim.now.V_den # + sim.now.V_pov

"""
    update_V_af!(sim::Simulation)

Update the volume contribution from ice above floatation as in [goelzer-brief-2020](@cite) (eq. 13).
Note: we do not use (eq. 1) as it is only a special case of (eq. 13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(sim::Simulation)
    sim.tools.prealloc.buffer_x .= (sim.now.H_af .- sim.ref.H_af) .* sim.domain.A 
    sim.now.V_af = sum(sim.tools.prealloc.buffer_x)
    return nothing
end

"""
    update_V_den!(sim::Simulation)

Update the volume contribution associated with the density difference between meltwater and
sea water, as in [goelzer-brief-2020](@cite) (eq. 10).
"""
function update_V_den!(sim::Simulation)
    density_factor = sim.c.rho_ice / sim.c.rho_water - sim.c.rho_ice / sim.c.rho_seawater
    sim.tools.prealloc.buffer_x .= (sim.now.H_water .- sim.ref.H_water) .* sim.domain.A
    sim.now.V_den = sum( sim.tools.prealloc.buffer_x ) * density_factor
    return nothing
end

"""
    update_V_pov!(sim::Simulation)

Update the volume contribution to the ocean (from isostatic adjustement in ocean regions),
which corresponds to the "potential ocean volume" in [goelzer-brief-2020](@cite) (eq. 14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(sim::Simulation)
    sim.tools.prealloc.buffer_x .= sim.now.z_b .- sim.now.z_ss
    sim.tools.prealloc.buffer_x .= max.(sim.tools.prealloc.buffer_x, 0) .* sim.domain.A
    sim.tools.prealloc.buffer_x .= sim.tools.prealloc.buffer_x .* (sim.now.z_b .< sim.now.z_ss)
    sim.now.V_pov = sum( sim.tools.prealloc.buffer_x )
    return nothing
end