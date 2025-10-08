# AbstractBSLcontributions
"""
$(TYPEDSIGNATURES)

An abstract type for barystatic sea level contributions. Subtypes are:
 - [`AbstractVolumeContribution`](@ref)
 - [`AbstractDensityContribution`](@ref)
 - [`AbstractAdjustmentContribution`](@ref)

This is typically used during the initialisation of a [`RegionalSeaLevel`](@ref) struct. For instance, if we want to compute the volume and density contributions following [goelzer-brief-2020](@citet) and ignore the adjustment contribution, we would write:

```julia
sealevel = RegionalSeaLevel(
    volume_contribution = GoelzerVolumeContribution(),
    density_contribution = GoelzerDensityContribution(),
    adjustment_contribution = NoAdjustmentContribution()
)
```

"""
abstract type AbstractBarystaticContribution end

# AbstractVolumeContribution
"""
$(TYPEDSIGNATURES)

An abstract subtype of [`AbstractBarystaticContribution`] that accounts for the volume contribution. Subtypes are:
 - [`NoVolumeContribution`](@ref)
 - [`GoelzerVolumeContribution`](@ref)
"""
abstract type AbstractVolumeContribution end

"""
$(TYPEDSIGNATURES)

A struct to ignore volume contribution to barystatic sea level.
"""
struct NoVolumeContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the volume contribtion to barystatic sea level following [goelzer-brief-2020](@citet).
"""
struct GoelzerVolumeContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the volume contribtion to barystatic sea level following [adhikari-kinematic-2020](@citet).
"""
struct AdhikariVolumeContribution end


# AbstractDensityContribution
"""
$(TYPEDSIGNATURES)

An abstract subtype of [`AbstractBarystaticContribution`] that accounts for the density contribution. Subtypes are:
 - [`NoDensityContribution`](@ref)
 - [`GoelzerDensityContribution`](@ref)
"""
abstract type AbstractDensityContribution end

"""
$(TYPEDSIGNATURES)

A struct to ignore density contribution to barystatic sea level.
"""
struct NoDensityContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the density contribtion to barystatic sea level following [goelzer-brief-2020](@citet).
"""
struct GoelzerDensityContribution end

# AbstractAdjustmentContribution
"""
$(TYPEDSIGNATURES)

An abstract subtype of [`AbstractBarystaticContribution`] that accounts for the adjustment contribution. Subtypes are:
 - [`NoAdjustmentContribution`](@ref)
 - [`GoelzerAdjustmentContribution`](@ref)
"""
abstract type AbstractAdjustmentContribution end

"""
$(TYPEDSIGNATURES)

An empty struct indicating that there is no adjustment contribution to barystatic sea level.
"""
struct NoAdjustmentContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the adjustment contribtion to barystatic sea level following [goelzer-brief-2020](@citet).
"""
struct GoelzerAdjustmentContribution end

##################################################################
# Sea level
################################################################

"""
$(TYPEDSIGNATURES)

Abstract type for sea surface representation. Available subtypes are:
 - [`LaterallyConstantSeaSurface`](@ref)
 - [`LaterallyVariableSeaSurface`](@ref)
"""
abstract type AbstractSeaSurface end

"""
$(TYPEDSIGNATURES)

Assume a laterally constant sea surface across the domain. This means that
the gravitatiional response is ignored.
"""
struct LaterallyConstantSeaSurface <: AbstractSeaSurface end

"""
$(TYPEDSIGNATURES)

Assume a laterally variable sea surface across the domain. This means that
the gravitational response is included in the sea surface perturbation.
"""
struct LaterallyVariableSeaSurface <: AbstractSeaSurface end

##################################################################
# RegionalSeaLevel
################################################################

"""
$(TYPEDSIGNATURES)

A struct that gathers the modelling choices for the sea-level component of the simulation.
It contains:
 - `surface`: an instance of [`AbstractSeaSurface`](@ref) to represent the sea surface.
 - `load`: an instance of [`AbstractSealevelLoad`](@ref) to represent the sea-level load.
 - `bsl`: an instance of [`AbstractBSL`](@ref) to represent the barystatic sea level.
 - `update_bsl`: an instance of [`AbstractUpdateBSL`](@ref) to represent the update mechanism for the barystatic sea level.
"""
@kwdef struct RegionalSeaLevel{
    S,          # <:AbstractSeaSurface,
    L,          # <:AbstractSealevelLoad,
    BSL,        # <:AbstractBSL,
    UBSL,       # <:AbstractUpdateBSL,
    VC,         # <:AbstractVolumeContribution,
    AC,         # <:AbstractAdjustmentContribution,
    DC          # <:AbstractDensityContribution
}
    surface::S = LaterallyConstantSeaSurface()  # lc or lv
    load::L = NoSealevelLoad()                  # no or interactive
    bsl::BSL = ConstantBSL()                    # constant, imposed, pw-constant or -linear
    update_bsl::UBSL = InternalUpdateBSL()      # internal or external
    volume_contribution::VC = GoelzerVolumeContribution()
    density_contribution::DC = GoelzerDensityContribution()
    adjustment_contribution::AC = NoAdjustmentContribution()
end

"""
$(TYPEDSIGNATURES)

Update the SSH perturbation `dz_ss` by convoluting the Green's function with the load anom.
"""
function update_dz_ss!(sim::Simulation, sl::LaterallyVariableSeaSurface)

    # Assume that lithosphere carries without gravity anomaly (due to compression)
    @. sim.tools.prealloc.buffer_x = sim.now.columnanoms.load +
        sim.solidearth.maskactive * sim.now.columnanoms.mantle
    @. sim.tools.prealloc.buffer_x = mass_anom(sim.domain.A, sim.tools.prealloc.buffer_x)
    samesize_conv!(sim.now.dz_ss, sim.tools.prealloc.buffer_x,
        sim.tools.dz_ss_convo, sim.tools.conv_helpers,
        sim.domain, sim.bcs.sea_surface_perturbation,
        sim.bcs.sea_surface_perturbation.space)
    return nothing
end

function update_dz_ss!(sim::Simulation, sl::LaterallyConstantSeaSurface)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return the Green's function used to compute the SSH perturbation `dz_ss` as in [^Coulon2021].
"""
function get_dz_ss_green(domain::RegionalDomain, c::PhysicalConstants)
    dz_ssgreen = unbounded_dz_ssgreen(domain.R, c)
    # max_dz_ssgreen = unbounded_dz_ssgreen(norm([100e3, 100e3]), c)
        # tolerance = resolution on 100km
    max_dz_ssgreen = unbounded_dz_ssgreen(domain.dx/2, c)
    return min.(dz_ssgreen, max_dz_ssgreen)
    # equivalent to: dz_ssgreen[dz_ssgreen .> max_dz_ssgreen] .= max_dz_ssgreen
end

function unbounded_dz_ssgreen(R, c::PhysicalConstants)
    return c.r_pole ./ ( 2 .* c.mE .* sin.( R ./ (2 .* c.r_pole) ) )
end

"""
$(TYPEDSIGNATURES)

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
    update_bsl!(sim.sealevel.bsl, -sim.now.delta_V, sim.nout.t_steps_ode[end])
    sim.now.z_bsl = sim.sealevel.bsl.z
    return nothing
end

function internal_update_bsl!(sim::Simulation, up::ExternalUpdateBSL)
    update_delta_V!(sim)
    return nothing
end

function update_delta_V!(sim::Simulation)
    Vold = total_volume(sim)
    update_V_af!(sim, sim.sealevel.volume_contribution)
    update_V_den!(sim, sim.sealevel.density_contribution)
    update_V_pov!(sim, sim.sealevel.adjustment_contribution)
    Vnew = total_volume(sim)
    sim.now.delta_V = Vnew - Vold
    return nothing
end

total_volume(sim::Simulation) = sim.now.V_af + sim.now.V_den + sim.now.V_pov

"""
$(TYPEDSIGNATURES)

Update the volume contribution from ice above floatation as in [goelzer-brief-2020](@cite) (eq. 13).
Note: we do not use (eq. 1) as it is only a special case of (eq. 13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(sim::Simulation, vc::NoVolumeContribution)
    sim.now.V_af = 0.0
    return nothing
end

function update_V_af!(sim::Simulation, vc::GoelzerVolumeContribution)
    sim.tools.prealloc.buffer_x .= sim.now.H_af .* sim.domain.A 
    sim.now.V_af = sum(sim.tools.prealloc.buffer_x) * sim.c.rho_ice / sim.c.rho_seawater
    return nothing
end

function update_V_af!(sim::Simulation, vc::AdhikariVolumeContribution)
    L = 1 - mask_ocean
    Lp1 = 1 - mask_ocean_p1
    delta_H_m = delta_H * L * Lp1 + delta_H_f * (1 - L * Lp1)
    delta_H_v = (1 - rho_water / rho_seawater) * (delta_H - delta_H_f) * (1 - L * Lp1)
    sim.tools.prealloc.buffer_x .= (delta_H_m + delta_H_v) .* sim.domain.A
    sim.now.V_af = sum(sim.tools.prealloc.buffer_x)
end

"""
$(TYPEDSIGNATURES)

Update the volume contribution associated with the density difference between meltwater and
sea water, as in [goelzer-brief-2020](@cite) (eq. 10).
"""
function update_V_den!(sim::Simulation, dc::NoDensityContribution)
    sim.now.V_den = 0.0
    return nothing
end

function update_V_den!(sim::Simulation, dc::GoelzerDensityContribution)
    density_factor = sim.c.rho_ice / sim.c.rho_water - sim.c.rho_ice / sim.c.rho_seawater
    sim.tools.prealloc.buffer_x .= sim.now.H_ice .* sim.domain.A
    sim.now.V_den = sum( sim.tools.prealloc.buffer_x ) * density_factor
    return nothing
end


"""
$(TYPEDSIGNATURES)

Update the volume contribution to the ocean (from isostatic adjustement in ocean regions),
which corresponds to the "potential ocean volume" in [goelzer-brief-2020](@cite) (eq. 14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(sim::Simulation, ac::NoAdjustmentContribution)
    sim.now.V_pov = 0.0
    return nothing
end

function update_V_pov!(sim::Simulation, ac::GoelzerAdjustmentContribution)
    # essentially watercolumn * surface
    sim.tools.prealloc.buffer_x .= sim.now.z_ss .- sim.now.z_b
    sim.tools.prealloc.buffer_x .= max.(sim.tools.prealloc.buffer_x, 0) .* sim.domain.A

    sim.now.V_pov = sum( sim.tools.prealloc.buffer_x )
    return nothing
end