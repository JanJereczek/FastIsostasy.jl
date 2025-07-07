###############################################################################
# Ice thickness BC
###############################################################################

"""
    AbstractIceThickness

An abstract type that determines how the ice thickness is updated in the model.
This is done by implementing the `update_ice!` function for different subtypes.
Available subtypes are:
- [`TimeInterpolatedIceThickness`](@ref)
- [`ExternallyUpdatedIceThickness`](@ref)
"""
abstract type AbstractIceThickness end

"""
    TimeInterpolatedIceThickness

A struct to update the ice thickness based on time interpolation.
Contains:
- `t_vec`: a vector of time points at which the ice thickness is defined.
- `H_vec`: a vector of ice thickness values corresponding to `t_vec`.
- `H_itp`: a function that interpolates the ice thickness based on time.
"""
struct TimeInterpolatedIceThickness{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    I<:TimeInterpolation2D{T, M},
} <: AbstractIceThickness

    t_vec::Vector{T}
    H_vec::Vector{M}
    H_itp::I
end

function TimeInterpolatedIceThickness(t_vec, H_vec, domain::RegionalDomain)
    H_vec = kernelpromote(H_vec, domain.arraykernel)
    itp = TimeInterpolation2D(t_vec, H_vec)
    return TimeInterpolatedIceThickness(t_vec, H_vec, itp)
end

"""
    ExternallyUpdatedIceThickness

A struct to indicate that the ice thickness is updated externally,
without any internal update.
"""
struct ExternallyUpdatedIceThickness <: AbstractIceThickness end

"""
    update_ice!(H, t, it::AbstractIceThickness)

Update the ice thickness `H` at time `t` using the method defined in `it`.
"""
function apply_bc!(H, t, it::TimeInterpolatedIceThickness)
    interpolate!(H, t, it.H_itp)
    return nothing
end

function apply_bc!(H, t, it::ExternallyUpdatedIceThickness)
    return nothing
end

###############################################################################
# Sediment BC
###############################################################################

abstract type AbstractSedimentThickness end

struct ExternallyUpdatedSedimentThickness <: AbstractSedimentThickness end
struct TimeInterpolatedSedimentThickness <: AbstractSedimentThickness end

###############################################################################
# Lateral BCs
###############################################################################

"""
    AbstractBCSpace

An abstract type representing the space in which boundary conditions are defined.
This typically needs to be defined when initializing an [`AbstractBCSpace`](@ref).
Available subtypes are:
- [`RegularBCSpace`](@ref)
- [`ExtendedBCSpace`](@ref)
"""
abstract type AbstractBCSpace end

"""
    RegularBCSpace <: AbstractBCSpace

Singleton struct to impose boundary conditions at the edges of the computation domain.
"""
struct RegularBCSpace <: AbstractBCSpace end

"""
    ExtendedBCSpace <: AbstractBCSpace

Singleton struct to impose boundary conditions at the edges of the extended
computation domain, which naturally arises from convolutions.
"""
struct ExtendedBCSpace <: AbstractBCSpace end

#########################################################################
# Computation-level
#########################################################################

"""
    AbstractBC

An abstract type representing a boundary condition in the context of a computational domain.
Available subtypes are:
- [`OffsetBC`](@ref)
"""
abstract type AbstractBC end

"""
    OffsetBC{T} <: AbstractBC

A boundary condition that applies an offset to the values at the boundaries of a
computational domain. Contains:
- `space`: the [`AbstractBCSpace`](@ref) in which the boundary condition is defined.
- `x_border`: the offset value to be applied at the boundaries.
- `W`: a weight matrix to apply the boundary condition according to some [ÀbstractBCRule](@ref).
"""
struct OffsetBC{T, M<:KernelMatrix{T}} <: AbstractBC
    space::AbstractBCSpace
    x_border::T
    W::M
    buffer::M
end

struct NoBC end

"""
    apply_bc!(X, bc::OffsetBC)

Apply the boundary condition `bc` to the matrix `X` in-place.
"""
function apply_bc!(X, bc::OffsetBC)
    bc.buffer .= bc.W .* X
    X .-= (sum(bc.buffer) - bc.x_border)
    return nothing
end

function apply_bc!(X, bc::NoBC)
    return nothing
end

#########################################################################
# API level
#########################################################################

"""
    CornerBC

Impose a Dirichlet-like boundary condition at the corners of the computational domain.
"""
struct CornerBC
    space::AbstractBCSpace
    x_border
end

"""
    BorderBC

Impose a Dirichlet-like boundary condition at the borders of the computational domain.
"""
struct BorderBC
    space::AbstractBCSpace
    x_border
end

"""
    DistanceWeightedBC

Impose a Dirichlet-like boundary condition at the borders of the computational domain,
weighted by the distance from the center of the domain.
"""
struct DistanceWeightedBC
    space::AbstractBCSpace
    x_border
end

"""
    MeanBC

Impose a mean value for the field.
"""
struct MeanBC
    space::AbstractBCSpace
    x_border
end

function corner_ones(T, nx, ny)
    W = zeros(T, nx, ny)
    for i in [1, nx]
        for j in [1, ny]
            W[i, j] = 1
        end
    end
    return W
end

function border_ones(T, nx, ny)
    W = zeros(T, nx, ny)
    for i in 1:nx
        for j in 1:ny
            if i == 1 || i == nx || j == 1 || j == ny
                W[i, j] = 1
            end
        end
    end
    return W
end

function norm!(W)
    W .= W ./ sum(W)
    return nothing
end

"""
    precompute_bc(bc::AbstractBC, sp::AbstractBCSpace, domain::RegionalDomain)

Precompute the boundary condition for the given computation domain.
"""
function precompute_bc(bc::CornerBC, sp::RegularBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = corner_ones(T, domain.nx, domain.ny)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

function precompute_bc(bc::CornerBC, sp::ExtendedBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = corner_ones(T, 2*domain.nx-1, 2*domain.ny-1)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

function precompute_bc(bc::BorderBC, sp::RegularBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = border_ones(T, domain.nx, domain.ny)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

function precompute_bc(bc::BorderBC, sp::ExtendedBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = border_ones(T, 2*domain.nx-1, 2*domain.ny-1)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

function precompute_bc(bc::DistanceWeightedBC, sp::RegularBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = border_ones(T, domain.nx, domain.ny)
    W .= W .* domain.R
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

function precompute_bc(bc::DistanceWeightedBC, sp::ExtendedBCSpace, domain::RegionalDomain)
    error("DistanceWeightedBC is not implemented for ExtendedBCSpace")
end

function precompute_bc(bc::MeanBC, sp::RegularBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = ones(T, domain.nx, domain.ny)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

function precompute_bc(bc::MeanBC, sp::ExtendedBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    W = ones(T, 2*domain.nx-1, 2*domain.ny-1)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, domain.arraykernel(W), similar(W))
end

#########################################################################
# Simulation level
#########################################################################

"""
    BoundaryConditions(viscous_displacement, elastic_displacement, geoid_perturbation, domain)

Create a `BoundaryConditions` struct containing the boundary conditions for the problem.
"""
struct BoundaryConditions{
    T,      # <:AbstractFloat,
    M,      # <:KernelMatrix{T},
    IT,     # <:AbstractIceThickness,
}
    ice_thickness::IT
    viscous_displacement::OffsetBC{T, M}
    elastic_displacement::OffsetBC{T, M}
    sea_surface_perturbation::OffsetBC{T, M}
end

function BoundaryConditions(
    domain::RegionalDomain{T, L, M};
    ice_thickness = ExternallyUpdatedIceThickness(),
    viscous_displacement = CornerBC(RegularBCSpace(), T(0)),
    elastic_displacement = CornerBC(ExtendedBCSpace(), T(0)),
    sea_surface_perturbation = CornerBC(ExtendedBCSpace(), T(0))) where
    {T<:AbstractFloat, L, M}
    
    # viscous_displacement must be defined on a regular grid
    @assert isa(viscous_displacement.space, RegularBCSpace)
    
    return BoundaryConditions(
        ice_thickness,
        precompute_bc(viscous_displacement, viscous_displacement.space, domain),
        precompute_bc(elastic_displacement, elastic_displacement.space, domain),
        precompute_bc(sea_surface_perturbation, sea_surface_perturbation.space, domain),
    )
end