##############################################################
# Lithosphere
##############################################################

"""
    AbstractLithosphere

Available subtypes are:
- [`RigidLithosphere`](@ref)
- [`LaterallyConstantLithosphere`](@ref)
- [`LaterallyVariableLithosphere`](@ref)
"""
abstract type AbstractLithosphere end

"""
    RigidLithosphere

Assume a rigid lithosphere, i.e. the elastic deformation is neglected.
"""
struct RigidLithosphere <: AbstractLithosphere end

"""
    LaterallyConstantLithosphere

Assume a laterally constant lithospheric thickness (and rigidity) across the domain.
This generally improves the performance of the solver, but is less realistic.
"""
struct LaterallyConstantLithosphere <: AbstractLithosphere end

"""
    LaterallyVariableLithosphere

Assume a laterally variable lithospheric thickness (and rigidity) across the domain.
This generally improves the realism of the model, but is more computationally expensive.
"""
struct LaterallyVariableLithosphere <: AbstractLithosphere end

##############################################################
# Mantle
##############################################################

"""
    AbstractMantle

Available subtypes are:
- [`RigidMantle`](@ref)
- [`LaterallyConstantMantle`](@ref)
- [`LaterallyVariableMantle`](@ref)
"""
abstract type AbstractMantle end

"""
    RigidMantle

Assume a rigid mantle, i.e. the viscous/relaxed deformation is neglected.
"""
struct RigidMantle <: AbstractMantle end

"""
    LaterallyConstantMantle

Assume a laterally-constant mantle properties (relaxation time or viscosity
depending on the [`AbstractRheology`](@ref) that is used) across the domain.
This generally improves the performance of the solver, but is less realistic.
"""
struct LaterallyConstantMantle <: AbstractMantle end

"""
    LaterallyVariableMantle

Assume a laterally-variable mantle properties (relaxation time or viscosity
depending on the [`AbstractRheology`](@ref) that is used) across the domain.
This generally improves the realism of the model, but is more computationally expensive.
"""
struct LaterallyVariableMantle <: AbstractMantle end

##############################################################
# Rheology
##############################################################

"""
    AbstractRheology

Available subtypes are:
- [`RelaxedRheology`](@ref)
- [`MaxwellRheology`](@ref)
"""
abstract type AbstractRheology end

"""
    RelaxedRheology

Assume a relaxed rheology in the mantle, governed by a relaxation time.
This generally offers worse performance and is less realistic than a viscous rheology.
It is only included for legacy purpose (e.g. comparison among solvers).
"""
struct RelaxedRheology <: AbstractRheology end

"""
    MaxwellRheology

Assume a viscous rheology in the mantle, governed by a viscosity.
This is the most realistic rheology and generally offers the best performance.
It is the default rheology used in the solver.
"""
struct MaxwellRheology <: AbstractRheology end

"""
    BurgersRheology

Assume a Burgers rheology in the mantle, which is a combination of a Maxwell and a Kelvin-Voigt rheology.
This is a more complex rheology that can capture both short-term and long-term viscoelastic behavior.
It is not yet implemented in the solver.
"""
struct BurgersRheology <: AbstractRheology end

##################################################################
# EarthModel
################################################################

struct EarthModel{L<:AbstractLithosphere, M<:AbstractMantle, R<:AbstractRheology}
    lithosphere::L      # lc or lv
    mantle::M           # lc or lv
    rheology::R         # relaxed, maxwell or burgers
end

# lc, lc, relaxed = ELRA
# lv, lv, relaxed = LVELRA
# lc, lc, viscous = ELVA
# lv, lv, viscous = LVELVA

# But we could also do:
# lc, lv, relaxed => ELRA solver
# lv, lc, relaxed => LVELRA solver
# lc, lv, viscous => ELVA solver
# lv, lc, viscous => LVELVA solver