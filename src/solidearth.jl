##############################################################
# Lithosphere
##############################################################

"""
$(TYPEDSIGNATURES)

Available subtypes are:
- [`RigidLithosphere`](@ref)
- [`LaterallyConstantLithosphere`](@ref)
- [`LaterallyVariableLithosphere`](@ref)
"""
abstract type AbstractLithosphere end

"""
$(TYPEDSIGNATURES)

Assume a rigid lithosphere, i.e. the elastic deformation is neglected.
"""
struct RigidLithosphere <: AbstractLithosphere end

"""
$(TYPEDSIGNATURES)

Assume a laterally constant lithospheric thickness (and rigidity) across the domain.
This generally improves the performance of the solver, but is less realistic.
"""
struct LaterallyConstantLithosphere <: AbstractLithosphere end

"""
$(TYPEDSIGNATURES)

Assume a laterally variable lithospheric thickness (and rigidity) across the domain.
This generally improves the realism of the model, but is more computationally expensive.
"""
struct LaterallyVariableLithosphere <: AbstractLithosphere end

##############################################################
# Mantle
##############################################################

"""
$(TYPEDSIGNATURES)

Available subtypes are:
- [`RigidMantle`](@ref)
- [`RelaxedMantle`](@ref)
- [`MaxwellMantle`](@ref)
"""
abstract type AbstractMantle end

"""
$(TYPEDSIGNATURES)

Assume a rigid mantle that does not deform.
"""
struct RigidMantle <: AbstractMantle end

"""
$(TYPEDSIGNATURES)

Assume a relaxed mantle that deforms according to a relaxation time.
This is generally less realistic and offers worse performance than a viscous mantle.
It is only included for legacy purpose (e.g. comparison among solvers).
"""
struct RelaxedMantle <: AbstractMantle end

"""
$(TYPEDSIGNATURES)

Assume a viscous mantle that deforms according to a viscosity.
This is the most realistic mantle model and generally offers the best performance.
It is the default mantle model used in the solver.
"""
struct MaxwellMantle <: AbstractMantle end

"""
$(TYPEDSIGNATURES)

Not implemented yet!
"""
struct BurgersMantle <: AbstractMantle end

###############################################################
# Solid Earth
###############################################################

const DEFAULT_RHO_LITHO = 3.2e3
const DEFAULT_LITHO_YOUNGMODULUS = 6.6e10
const DEFAULT_LITHO_POISSONRATIO = 0.28
const DEFAULT_LITHO_THICKNESS = 88e3
const DEFAULT_RHO_UPPERMANTLE = 3.4e3
const DEFAULT_MANTLE_POISSONRATIO = 0.28
const DEFAULT_MANTLE_TAU = 855.0

"""
$(TYPEDSIGNATURES)

Return a struct containing all information related to the lateral variability of
solid-Earth parameters. To initialize with values other than default, run:

```julia
domain = RegionalDomain(3000e3, 7)
lb = [100e3, 300e3]
lv = [1e19, 1e21]
solidearth = SolidEarth(domain, layer_boundaries = lb, layer_viscosities = lv)
```

which initializes a lithosphere of thickness ``T_1 = 100 \\mathrm{km}``, a viscous
channel between ``T_1``and ``T_2 = 300 \\mathrm{km}``and a viscous halfspace starting
at ``T_2``. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `domain::RegionalDomain`.
"""
mutable struct SolidEarth{
    T,  # <:AbstractFloat,
    M,  # <:KernelMatrix{T},
    B,  # <:KernelMatrix{Bool},
    LI, # <:AbstractLithosphere,
    MA, # <:AbstractMantle,
    CA, # <:AbstractCalibration,
    CO, # <:AbstractCompressibility,
    LU, # <:AbstractLumping,
}
    lithosphere::LI
    mantle::MA
    calibration::CA
    compressibility::CO
    lumping::LU
    effective_viscosity::M
    pseudodiff_scaling::M
    scaled_pseudodiff_inv::M
    litho_thickness::M
    litho_rigidity::M
    maskactive::B
    litho_poissonratio::T
    mantle_poissonratio::T
    tau::M
    scale_elralength::T
    litho_youngmodulus::T
    litho_shearmodulus::T
    rho_uppermantle::T
    rho_litho::T
end

function SolidEarth(
    domain::RegionalDomain{T, L, M};
    lithosphere = LaterallyVariableLithosphere(),
    mantle = MaxwellMantle(),
    calibration = NoCalibration(),
    compressibility = CompressibleMantle(),
    lumping = FreqDomainViscosityLumping(),
    maskactive = domain.R .< Inf,
    layer_boundaries = T.([88e3, 400e3]),
    layer_viscosities = T.([1e19, 1e21]),           # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
    litho_youngmodulus = T(DEFAULT_LITHO_YOUNGMODULUS),              # (N/m^2)
    litho_poissonratio = T(DEFAULT_LITHO_POISSONRATIO),
    mantle_poissonratio = T(DEFAULT_MANTLE_POISSONRATIO),
    tau = T(DEFAULT_MANTLE_TAU),
    scale_elralength = T(1),                        # Following LeMeur (1996, text below Eq. 3)
    rho_uppermantle = T(DEFAULT_RHO_UPPERMANTLE),   # Mean density of topmost upper mantle (kg m^-3)
    rho_litho = T(DEFAULT_RHO_LITHO),               # Mean density of lithosphere (kg m^-3)
) where {T<:AbstractFloat, L, M}

    if tau isa Real
        tau = fill(tau, domain)
    end
    tau = kernelpromote(tau, domain.arraykernel)

    if layer_boundaries isa Vector
        layer_boundaries = matrify(layer_boundaries, domain.nx, domain.ny)
    end

    if layer_viscosities isa Vector
        layer_viscosities = matrify(layer_viscosities, domain.nx, domain.ny)
    end

    litho_thickness = zeros(T, domain.nx, domain.ny)
    litho_thickness .= view(layer_boundaries, :, :, 1)

    litho_rigidity = get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)
    effective_viscosity, pseudodiff_scaling = get_effective_viscosity_and_scaling(
        domain, layer_viscosities, layer_boundaries, maskactive, lumping)

    apply_compressibility!(effective_viscosity, mantle_poissonratio, compressibility)
    apply_calibration!(effective_viscosity, calibration)

    litho_thickness, litho_rigidity, effective_viscosity, pseudodiff_scaling, maskactive =
        kernelpromote( [litho_thickness, litho_rigidity, effective_viscosity,
        pseudodiff_scaling, maskactive], domain.arraykernel)

    scaled_pseudodiff_inv = 1 ./ (pseudodiff_scaling .* domain.pseudodiff)

    litho_shearmodulus = get_shearmodulus(litho_youngmodulus, litho_poissonratio)

    return SolidEarth(
        lithosphere, mantle, calibration, compressibility, lumping,
        effective_viscosity, pseudodiff_scaling, scaled_pseudodiff_inv,
        litho_thickness, litho_rigidity, kernelcollect(maskactive, domain),
        litho_poissonratio, mantle_poissonratio, tau, scale_elralength,
        litho_youngmodulus, litho_shearmodulus, rho_uppermantle, rho_litho,
    )

end