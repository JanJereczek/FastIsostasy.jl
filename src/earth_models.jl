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
- [`RelaxedMantle`](@ref)
- [`MaxwellMantle`](@ref)
"""
abstract type AbstractMantle end

"""
    RigidMantle

Assume a rigid mantle that does not deform.
"""
struct RigidMantle <: AbstractMantle end

"""
    RelaxedMantle

Assume a relaxed mantle that deforms according to a relaxation time.
This is generally less realistic and offers worse performance than a viscous mantle.
It is only included for legacy purpose (e.g. comparison among solvers).
"""
struct RelaxedMantle <: AbstractMantle end

"""
    MaxwellMantle

Assume a viscous mantle that deforms according to a viscosity.
This is the most realistic mantle model and generally offers the best performance.
It is the default mantle model used in the solver.
"""
struct MaxwellMantle <: AbstractMantle end

"""
    BurgersMantle

Not implemented yet!
"""
struct BurgersMantle <: AbstractMantle end

##################################################################
# SolidEarthModel
################################################################

@kwdef struct SolidEarthModel{L<:AbstractLithosphere, M<:AbstractMantle}
    lithosphere::L = LaterallyVariableLithosphere()     # rigid, lc or lv
    mantle::M = MaxwellMantle()                         # rigid, relaxed or maxwell
end