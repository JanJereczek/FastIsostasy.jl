"""
$(TYPEDSIGNATURES)

An abstract type for the individual terms of the [goelzer_brief_2020](@citet)
decomposition of the barystatic sea level. Subtypes are:
 - [`AbstractVolumeContribution`](@ref)
 - [`AbstractDensityContribution`](@ref)
 - [`AbstractAdjustmentContribution`](@ref)

The three of them are selected together through a [`GoelzerBSLFormalism`](@ref), which
is what [`RegionalSeaLevel`](@ref) stores. For instance, to compute the volume and
density contributions and ignore the adjustment contribution (the default):

```julia
sealevel = RegionalSeaLevel(formalism = GoelzerBSLFormalism(
    volume = GoelzerVolumeContribution(),
    density = GoelzerDensityContribution(),
    adjustment = NoAdjustmentContribution(),
))
```

"""
abstract type AbstractBarystaticContribution end

# AbstractVolumeContribution
"""
$(TYPEDSIGNATURES)

An abstract subtype of [`AbstractBarystaticContribution`](@ref) that accounts for the volume contribution. Subtypes are:
 - [`NoVolumeContribution`](@ref)
 - [`GoelzerVolumeContribution`](@ref)
"""
abstract type AbstractVolumeContribution <: AbstractBarystaticContribution end

"""
$(TYPEDSIGNATURES)

A struct to ignore volume contribution to barystatic sea level.
"""
struct NoVolumeContribution <: AbstractVolumeContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the volume contribtion to barystatic sea level following [goelzer_brief_2020](@citet).
"""
struct GoelzerVolumeContribution <: AbstractVolumeContribution end

# AbstractDensityContribution
"""
$(TYPEDSIGNATURES)

An abstract subtype of [`AbstractBarystaticContribution`](@ref) that accounts for the density contribution. Subtypes are:
 - [`NoDensityContribution`](@ref)
 - [`GoelzerDensityContribution`](@ref)
"""
abstract type AbstractDensityContribution <: AbstractBarystaticContribution end

"""
$(TYPEDSIGNATURES)

A struct to ignore density contribution to barystatic sea level.
"""
struct NoDensityContribution <: AbstractDensityContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the density contribtion to barystatic sea level following [goelzer_brief_2020](@citet).
"""
struct GoelzerDensityContribution <: AbstractDensityContribution end

# AbstractAdjustmentContribution
"""
$(TYPEDSIGNATURES)

An abstract subtype of [`AbstractBarystaticContribution`](@ref) that accounts for the adjustment contribution. Subtypes are:
 - [`NoAdjustmentContribution`](@ref)
 - [`GoelzerAdjustmentContribution`](@ref)
"""
abstract type AbstractAdjustmentContribution <: AbstractBarystaticContribution end

"""
$(TYPEDSIGNATURES)

An empty struct indicating that there is no adjustment contribution to barystatic sea level.
"""
struct NoAdjustmentContribution <: AbstractAdjustmentContribution end

"""
$(TYPEDSIGNATURES)

A struct to compute the adjustment contribtion to barystatic sea level following [goelzer_brief_2020](@citet).
"""
struct GoelzerAdjustmentContribution <: AbstractAdjustmentContribution end

################################################################
# BSL formalisms
################################################################

"""
$(TYPEDSIGNATURES)

Abstract type selecting **how** the barystatic sea-level (BSL) contribution of the
domain is computed. Available subtypes are:
 - [`GoelzerBSLFormalism`](@ref) (default)
 - [`AdhikariBSLFormalism`](@ref)

The two are alternative *decompositions* of the same quantity, not alternative
terms of one decomposition, which is why they sit on a single axis of
[`RegionalSeaLevel`](@ref) rather than being mixed term by term.
"""
abstract type AbstractBSLFormalism end

"""
$(TYPEDSIGNATURES)

Compute the BSL contribution as in [goelzer_brief_2020](@citet): three *absolute*
volumes — ice above floatation, meltwater/seawater density correction and
potential ocean volume — recomputed at every coupling interval and differenced
against their previous values.

# Fields
 - `volume`: an [`AbstractVolumeContribution`](@ref), `V_af`.
 - `density`: an [`AbstractDensityContribution`](@ref), `V_den`.
 - `adjustment`: an [`AbstractAdjustmentContribution`](@ref), `V_pov`.

```jldoctest
julia> using FastIsostasy

julia> GoelzerBSLFormalism()
GoelzerBSLFormalism{GoelzerVolumeContribution, GoelzerDensityContribution, NoAdjustmentContribution}(GoelzerVolumeContribution(), GoelzerDensityContribution(), NoAdjustmentContribution())
```
"""
@kwdef struct GoelzerBSLFormalism{VC,DC,AC} <: AbstractBSLFormalism
    volume::VC = GoelzerVolumeContribution()
    density::DC = GoelzerDensityContribution()
    adjustment::AC = NoAdjustmentContribution()
end

"""
$(TYPEDSIGNATURES)

Compute the BSL contribution with the kinematic formalism of
[adhikari_kinematic_2020](@citet): one *incremental* per-cell field
`ΔH_S = ΔH_M + ΔH_V` over each coupling interval `Δt` (their Eqs. 10–12),

    ΔH_M = ΔH ⋅ ℒ(t)ℒ(t+Δt)  +  ΔH_F ⋅ [1 - ℒ(t)ℒ(t+Δt)]
    ΔH_V = (1 - ρ_w/ρ_o) ⋅ [ΔH - ΔH_F] ⋅ [1 - ℒ(t)ℒ(t+Δt)]

with `ΔH` the ice-thickness change, `ΔH_F` the change in height above floatation
(see [`update_HF!`](@ref)) and `ℒ` the land mask. `ΔH_M` changes ocean mass *and*
volume, `ΔH_V` changes ocean volume only — meltwater occupies more volume than the
seawater it displaces. The land-mask product is 1 only where the cell was land at
*both* ends of the interval (their Regime 1, where all of `ΔH` counts) and 0
wherever it transitioned or stayed ocean (Regimes 2 and 3, where the change in
height above floatation counts instead).

Unlike [`GoelzerBSLFormalism`](@ref), the conversion to sea-level equivalent is done in
**freshwater** throughout, `ρ_i/ρ_w`, with the leftover density difference folded
into `ΔH_V`; the two decompositions are therefore not comparable term by term,
only in their sum.

Selecting this formalism activates the two-time-level buffers
[`KinematicBSL`](@ref) on [`CurrentState`](@ref).

```jldoctest
julia> using FastIsostasy

julia> RegionalSeaLevel(formalism = AdhikariBSLFormalism())
 Sea surface:         LaterallyConstantSeaSurface
 Sea-level load:      NoSealevelLoad
 Barystatic sea level: ConstantBSL{Float32, ReferenceBSL{Float32, TimeInterpolation0D{Float32}}}
 BSL update:          InternalBSLUpdate
 BSL formalism:       AdhikariBSLFormalism
```

!!! note "Two caveats of the current implementation"
    (i) The ocean mask is evaluated pointwise, not repaired for connectivity to the
    global ocean (their Eq. 3), so isolated below-floatation regions count as ocean
    and `H_F` never goes negative.
    (ii) The ocean area dividing the freshwater volume is the global hypsometric
    table of [`ReferenceBSL`](@ref) rather than their `A_ocean(t+Δt)` — a
    regional-model stand-in. Their Eq. (14) mass-conserving load is likewise not
    applied; the ice load still uses the full column anomaly.
"""
struct AdhikariBSLFormalism <: AbstractBSLFormalism end

# Only `AdhikariBSLFormalism` is two-time-level, so only it needs the buffers of
# `KinematicBSL`; every other formalism gets them zero-sized. Queried by the
# `Simulation` constructor, mirroring `nbranches(mantle)`.
needs_kinematic_state(::AbstractBSLFormalism) = false
needs_kinematic_state(::AdhikariBSLFormalism) = true

################################################################
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

struct ImposedSeaSurface{ITP} <: AbstractSeaSurface
    dz_ss_itp::ITP
end

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
 - `update_bsl`: an instance of [`AbstractBSLUpdate`](@ref) to represent the update mechanism for the barystatic sea level.
 - `formalism`: an instance of [`AbstractBSLFormalism`](@ref) deciding how the BSL
   contribution of the domain is computed, [`GoelzerBSLFormalism`](@ref) (default) or
   [`AdhikariBSLFormalism`](@ref).

```jldoctest
julia> using FastIsostasy

julia> RegionalSeaLevel()
 Sea surface:         LaterallyConstantSeaSurface
 Sea-level load:      NoSealevelLoad
 Barystatic sea level: ConstantBSL{Float32, ReferenceBSL{Float32, TimeInterpolation0D{Float32}}}
 BSL update:          InternalBSLUpdate
 BSL formalism:       GoelzerBSLFormalism{GoelzerVolumeContribution, GoelzerDensityContribution, NoAdjustmentContribution}
```

The `volume_contribution`, `density_contribution` and `adjustment_contribution`
keywords are deprecated: they are the three terms of a single decomposition and
now live inside [`GoelzerBSLFormalism`](@ref). Passing them still works and builds the
corresponding `formalism`, overriding whatever `formalism` was given.
"""
struct RegionalSeaLevel{
    S,          # <:AbstractSeaSurface,
    L,          # <:AbstractSealevelLoad,
    BSL,        # <:AbstractBSL,
    UBSL,       # <:AbstractBSLUpdate,
    F,          # <:AbstractBSLFormalism
}
    surface::S          # lc or lv
    load::L             # no or interactive
    bsl::BSL            # constant, imposed, pw-constant or -linear
    update_bsl::UBSL    # internal or external
    formalism::F        # Goelzer or Adhikari
end

function RegionalSeaLevel(;
    surface = LaterallyConstantSeaSurface(),
    load = NoSealevelLoad(),
    bsl = ConstantBSL(),
    update_bsl = InternalBSLUpdate(),
    formalism = GoelzerBSLFormalism(),
    volume_contribution = nothing,
    density_contribution = nothing,
    adjustment_contribution = nothing,
)
    # The deprecated keywords win over `formalism` if both are given: they can
    # only ever describe a `GoelzerBSLFormalism`, so honouring them is the reading
    # that keeps an old script doing what it used to do.
    goelzer_kwargs = (volume_contribution, density_contribution, adjustment_contribution)
    if any(!isnothing, goelzer_kwargs)
        Base.depwarn(
            "The `volume_contribution`, `density_contribution` and " *
            "`adjustment_contribution` keywords of `RegionalSeaLevel` are " *
            "deprecated. Use `formalism = GoelzerBSLFormalism(volume = ..., " *
            "density = ..., adjustment = ...)`.",
            :RegionalSeaLevel,
        )
        formalism = GoelzerBSLFormalism(
            volume = something(volume_contribution, GoelzerVolumeContribution()),
            density = something(density_contribution, GoelzerDensityContribution()),
            adjustment = something(adjustment_contribution, NoAdjustmentContribution()),
        )
    end
    return RegionalSeaLevel(surface, load, bsl, update_bsl, formalism)
end

function Base.show(io::IO, ::MIME"text/plain", sl::RegionalSeaLevel)
    descriptors = [
        "Sea surface" => typeof(sl.surface),
        "Sea-level load" => typeof(sl.load),
        "Barystatic sea level" => typeof(sl.bsl),
        "BSL update" => typeof(sl.update_bsl),
        "BSL formalism" => typeof(sl.formalism),
    ]
    show_descriptors(io, descriptors)
end

"""
$(TYPEDSIGNATURES)

Update the SSH perturbation `dz_ss` by convoluting the Green's function with the load anom.
"""
function update_dz_ss!(sim::Simulation, sl::LaterallyVariableSeaSurface)

    # update_mass_anom! modifies sim.tools.prealloc.buffer_x in place
    update_mass_anom!(sim, sim.solidearth.lithosphere_column)
    samesize_conv!(
        sim.now.dz_ss,
        sim.tools.prealloc.buffer_x,
        sim.tools.dz_ss_convo,
        sim.tools.conv_helpers,
        sim.domain,
        sim.bcs.sea_surface_perturbation,
        sim.bcs.sea_surface_perturbation.space,
    )
    return nothing
end

function update_dz_ss!(sim::Simulation, sl::LaterallyConstantSeaSurface)
    return nothing
end

function update_dz_ss!(sim::Simulation, sl::ImposedSeaSurface)
    interpolate!(sim.now.dz_ss, sim.timer.t, sl.dz_ss_itp)
    return nothing
end

function update_mass_anom!(sim, lc::CompressibleLithosphereColumn)
    @. sim.tools.prealloc.buffer_x =
        sim.now.columnanoms.load +
        sim.solidearth.maskactive * sim.now.columnanoms.mantle
    @. sim.tools.prealloc.buffer_x =
        mass_anom(sim.domain.A, sim.tools.prealloc.buffer_x)
end

function update_mass_anom!(sim, lc::IncompressibleLithosphereColumn)
    @. sim.tools.prealloc.buffer_x = mass_anom(sim.domain.A, sim.now.columnanoms.full)
end

"""
$(TYPEDSIGNATURES)

Return the Green's function used to compute the SSH perturbation `dz_ss` as in
[coulon_contrasting_2021](@citet).
"""
function get_dz_ss_green(domain::RegionalDomain, c::PhysicalConstants)
    dz_ssgreen = unbounded_dz_ssgreen(domain.R, c)
    # Cap the singularity at the origin at the half-cell value.
    max_dz_ssgreen = unbounded_dz_ssgreen(domain.dx/2, c)
    return min.(dz_ssgreen, max_dz_ssgreen)
end

function unbounded_dz_ssgreen(R, c::PhysicalConstants)
    return c.r_pole ./ (2 .* c.mE .* sin.(R ./ (2 .* c.r_pole)))
end

"""
$(TYPEDSIGNATURES)

Update the sea-level by adding the various contributions as in [coulon_contrasting_2021](@cite).
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
Note that this differs from [goelzer_brief_2020](@cite) (eq. 12) because the ocean
surface is not assumed to be constant. Furthermore, the contribution to ocean volume
from the bedrock uplift is not included here since the volume displaced on site
is arguably blanaced by the depression of the peripherial forebulge.
"""
function internal_update_bsl!(sim::Simulation, up::InternalBSLUpdate)
    update_delta_V!(sim)
    update_bsl!(sim.sealevel.bsl, -sim.now.delta_V, sim.timer.t)
    sim.now.z_bsl = sim.sealevel.bsl.z
    return nothing
end

function internal_update_bsl!(sim::Simulation, up::ExternalBSLUpdate)
    update_delta_V!(sim)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update `sim.now.delta_V`, the volume (m³) withheld from the ocean over the coupling
interval that just elapsed, by dispatching on `sim.sealevel.formalism`. The sign
convention is fixed by `internal_update_bsl!`, which passes `-delta_V` to
[`update_bsl!`](@ref): a growing ice sheet gives `delta_V > 0` and a falling BSL.
"""
update_delta_V!(sim::Simulation) = update_delta_V!(sim, sim.sealevel.formalism)

function update_delta_V!(sim::Simulation, f::GoelzerBSLFormalism)
    Vold = total_volume(sim)
    update_V_af!(sim, f.volume)
    update_V_den!(sim, f.density)
    update_V_pov!(sim, f.adjustment)
    Vnew = total_volume(sim)
    sim.now.delta_V = Vnew - Vold
    return nothing
end

# Kinematic formalism of Adhikari et al. (2020), their Eqs. (10)-(12).
#
# Where Goelzer differences two absolute volumes, this differences the *fields*
# themselves against the values `sim.now.kinematic` holds from the previous
# coupling interval, and rolls that buffer forward at the end.
#
# The `H_af`/mask/`H_F` refresh at the top is what makes `ℒ(t)` and `ℒ(t+Δt)`
# comparable: on entry the masks are still the ones written at the end of the
# previous sparse-diagnostics block, i.e. consistent with the *old* ice thickness.
# Recomputing them here — before `z_ss` is updated further down the block — puts
# the `t+Δt` fields at exactly the point of the update sequence at which the `t`
# fields were captured, so the two ends of the interval come from one code path.
# `z_ss` therefore lags by one interval, exactly as it does in the Goelzer path
# (`update_V_af!` reads the same `H_af`); the *difference* over the interval is
# unaffected. The masks are recomputed again later in the block, so overwriting
# them here has no effect on the load.
function update_delta_V!(sim::Simulation, f::AdhikariBSLFormalism)
    now, k, c = sim.now, sim.now.kinematic, sim.c

    update_Haf!(sim)
    update_maskgrounded!(sim)
    update_maskocean!(sim)
    update_HF!(sim)

    # 1 - ρ_w/ρ_o: the volume by which meltwater exceeds the seawater it displaces.
    excess_volume = 1 - c.rho_water / c.rho_seawater

    # ΔH_M (their Eq. 11) and ΔH_V (their Eq. 12). `ℒ(t)ℒ(t+Δt)` is 1 only in
    # Regime 1 (land at both ends of the interval), where all of ΔH counts; its
    # complement selects Regimes 2 and 3, where ΔH_F counts instead.
    @. k.delta_H_M =
        (now.H_ice - k.H_ice_prev) * k.maskland_prev * not(now.maskocean) +
        (now.H_F - k.H_F_prev) * not(k.maskland_prev * not(now.maskocean))
    @. k.delta_H_V =
        excess_volume * ((now.H_ice - k.H_ice_prev) - (now.H_F - k.H_F_prev)) *
        not(k.maskland_prev * not(now.maskocean))

    # ΔGMSL = -(ρ_i/ρ_w) ∫ ΔH_S dA / A_ocean, and `update_bsl!` supplies the
    # division by the ocean area — so `delta_V` carries the freshwater-equivalent
    # volume *withheld* from the ocean, with no sign flip here.
    @. sim.tools.prealloc.buffer_x = (k.delta_H_M + k.delta_H_V) * sim.domain.A
    now.delta_V = totalsum(sim.tools.prealloc.buffer_x) * (c.rho_ice / c.rho_water)

    # Roll the interval forward: what is `t+Δt` here is `t` at the next call.
    k.H_ice_prev .= now.H_ice
    k.H_F_prev .= now.H_F
    @. k.maskland_prev = not(now.maskocean)
    return nothing
end

total_volume(sim::Simulation) = sim.now.V_af + sim.now.V_den + sim.now.V_pov

"""
$(TYPEDSIGNATURES)

Prime whatever state the BSL formalism differences against, so that the first
coupling interval is well defined. Called once by `init_problem!`.
"""
function init_bsl_formalism!(sim::Simulation, f::GoelzerBSLFormalism)
    update_V_af!(sim, f.volume)
    update_V_den!(sim, f.density)
    update_V_pov!(sim, f.adjustment)
    return nothing
end

function init_bsl_formalism!(sim::Simulation, f::AdhikariBSLFormalism)
    reset_kinematic!(sim.now.kinematic, sim.ref)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the volume contribution from ice above floatation as in [goelzer_brief_2020](@cite) (eq. 13).
Note: we do not use (eq. 1) as it is only a special case of (eq. 13) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_af!(sim::Simulation, vc::NoVolumeContribution)
    sim.now.V_af = 0
    return nothing
end

function update_V_af!(sim::Simulation, vc::GoelzerVolumeContribution)
    sim.tools.prealloc.buffer_x .= sim.now.H_af .* sim.domain.A
    sim.now.V_af =
        totalsum(sim.tools.prealloc.buffer_x) * sim.c.rho_ice / sim.c.rho_seawater
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the volume contribution associated with the density difference between meltwater and
sea water, as in [goelzer_brief_2020](@cite) (eq. 10).
"""
function update_V_den!(sim::Simulation, dc::NoDensityContribution)
    sim.now.V_den = 0
    return nothing
end

function update_V_den!(sim::Simulation, dc::GoelzerDensityContribution)
    density_factor =
        sim.c.rho_ice / sim.c.rho_water - sim.c.rho_ice / sim.c.rho_seawater
    sim.tools.prealloc.buffer_x .= sim.now.H_ice .* sim.domain.A
    sim.now.V_den = totalsum(sim.tools.prealloc.buffer_x) * density_factor
    return nothing
end


"""
$(TYPEDSIGNATURES)

Update the volume contribution to the ocean (from isostatic adjustement in ocean regions),
which corresponds to the "potential ocean volume" in [goelzer_brief_2020](@cite) (eq. 14).
Note: we do not use eq. (8) as it is only a special case of eq. (14) that does not
allow a correct representation of external sea-level forcings.
"""
function update_V_pov!(sim::Simulation, ac::NoAdjustmentContribution)
    sim.now.V_pov = 0
    return nothing
end

function update_V_pov!(sim::Simulation, ac::GoelzerAdjustmentContribution)
    # essentially watercolumn * surface
    sim.tools.prealloc.buffer_x .= sim.now.z_ss .- sim.now.z_b
    sim.tools.prealloc.buffer_x .= max.(sim.tools.prealloc.buffer_x, 0) .* sim.domain.A

    sim.now.V_pov = totalsum(sim.tools.prealloc.buffer_x)
    return nothing
end