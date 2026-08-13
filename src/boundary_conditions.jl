###############################################################################
# Ice thickness BC
###############################################################################

"""
$(TYPEDSIGNATURES)

Determine how the ice thickness is updated in the model by implementing `update_ice!`
for different subtypes.

# Available subtypes
- [`TimeInterpolatedIceThickness`](@ref)
- [`ExternallyUpdatedIceThickness`](@ref)
"""
abstract type AbstractIceThickness end

"""
$(TYPEDSIGNATURES)

Update the ice thickness based on a time interpolation.

# Fields
- `t_vec`: a vector of time points at which the ice thickness is defined.
- `H_vec`: a vector of ice thickness values corresponding to `t_vec`.
- `H_itp`: a function that interpolates the ice thickness based on time.
"""
struct TimeInterpolatedIceThickness{T,M,I<:TimeInterpolation2D} <: AbstractIceThickness
    t_vec::Vector{T}
    H_vec::Vector{M}
    H_itp::I
end

function TimeInterpolatedIceThickness(t_vec, H_vec, domain::RegionalDomain)
    H_vec = kernelpromote(H_vec, domain.backend)
    itp = TimeInterpolation2D(t_vec, H_vec)
    return TimeInterpolatedIceThickness(t_vec, H_vec, itp)
end

"""
$(TYPEDSIGNATURES)

Update the ice thickness externally, without any internal update.
"""
struct ExternallyUpdatedIceThickness <: AbstractIceThickness end

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

An abstract type representing the space in which boundary conditions are defined.
This typically needs to be defined when initializing an [`AbstractBCSpace`](@ref).

# Available subtypes
- [`RegularBCSpace`](@ref)
- [`ExtendedBCSpace`](@ref)
"""
abstract type AbstractBCSpace end

"""
$(TYPEDSIGNATURES)

Singleton struct to impose boundary conditions at the edges of the computation domain.
"""
struct RegularBCSpace <: AbstractBCSpace end

"""
$(TYPEDSIGNATURES)

Singleton struct to impose boundary conditions at the edges of the extended
computation domain, which naturally arises from convolutions.
"""
struct ExtendedBCSpace <: AbstractBCSpace end

#########################################################################
# Computation-level
#########################################################################

"""
$(TYPEDSIGNATURES)

An abstract type representing a boundary condition in the context of a computational domain.

# Available subtypes
- [`OffsetBC`](@ref)
- [`NoBC`](@ref)
"""
abstract type AbstractBC end

"""
$(TYPEDSIGNATURES)

Apply an offset to the values at the boundaries of a computational domain.

# Fields
- `space`: the [`AbstractBCSpace`](@ref) in which the boundary condition is defined.
- `x_border`: the offset value to be applied at the boundaries.
- `W`: a weight matrix to apply the boundary condition according to some [`AbstractBC`](@ref).
"""
struct OffsetBC{T,M} <: AbstractBC
    space::AbstractBCSpace
    x_border::T
    W::M
end

"""
$(TYPEDSIGNATURES)

A singleton struct representing the absence of a boundary condition.
"""
struct NoBC <: AbstractBC end

"""
$(TYPEDSIGNATURES)

Apply the boundary condition `bc` to the matrix `X` in-place.
"""
function apply_bc!(X, bc::OffsetBC)
    # `dot(bc.W, X) == sum(bc.W .* X)` for real arrays, computed allocation-free and
    # without mutating `bc` (so `bc` stays Enzyme-`Const`). Linear in-place op on `X`.
    X .-= (inner(bc.W, X) - bc.x_border)
    return nothing
end

function apply_bc!(X, bc::NoBC)
    return nothing
end

#########################################################################
# API level
#########################################################################

"""
$(TYPEDSIGNATURES)

Impose a Dirichlet-like boundary condition at the corners of the computational domain.
"""
struct CornerBC{B,T}
    space::B               # <:AbstractBCSpace
    x_border::T
end

"""
$(TYPEDSIGNATURES)

Impose a Dirichlet-like boundary condition at the borders of the computational domain.
"""
struct BorderBC{B,T}
    space::B               # <:AbstractBCSpace
    x_border::T
end

"""
$(TYPEDSIGNATURES)

Impose a Dirichlet-like boundary condition at the borders of the computational domain,
weighted by the distance from the center of the domain.
"""
struct DistanceWeightedBC{B,T}
    space::B               # <:AbstractBCSpace
    x_border::T
end


"""
$(TYPEDSIGNATURES)

Impose a mean value for the field.
"""
struct MeanBC{B,T}
    space::B               # <:AbstractBCSpace
    x_border::Any
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

function border_ones(T::Type{<:AbstractFloat}, nx::Integer, ny::Integer)
    W = zeros(T, nx, ny)
    for i = 1:nx
        for j = 1:ny
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

# The grid a BC's weights live on: the computation domain itself, or the larger
# one a convolution produces.
bc_gridsize(::RegularBCSpace, domain) = (domain.nx, domain.ny)
bc_gridsize(::ExtendedBCSpace, domain) = (2*domain.nx-1, 2*domain.ny-1)

# Unnormalised weights per BC flavour, on a grid of size `(nx, ny)`. `domain` is
# only needed by `DistanceWeightedBC`, which weights by distance from the centre
# and therefore has no `ExtendedBCSpace` counterpart.
bc_weights(::CornerBC, T, nx, ny, domain) = corner_ones(T, nx, ny)
bc_weights(::BorderBC, T, nx, ny, domain) = border_ones(T, nx, ny)
bc_weights(::MeanBC, T, nx, ny, domain) = ones(T, nx, ny)
bc_weights(::DistanceWeightedBC, T, nx, ny, domain) =
    border_ones(T, nx, ny) .* domain.R

"""
$(TYPEDSIGNATURES)

Precompute the boundary condition for the given computation domain, i.e. resolve
it into the normalised weight matrix `W` of an [`OffsetBC`](@ref).
"""
function precompute_bc(bc, sp::AbstractBCSpace, domain::RegionalDomain)
    T = eltype(domain.R)
    nx, ny = bc_gridsize(sp, domain)
    W = kernelpromote(bc_weights(bc, T, nx, ny, domain), domain.backend)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, W)
end

precompute_bc(bc::DistanceWeightedBC, sp::ExtendedBCSpace, domain::RegionalDomain) =
    error("DistanceWeightedBC is not implemented for ExtendedBCSpace")

#########################################################################
# Simulation level
#########################################################################

"""
$(TYPEDSIGNATURES)

Define the boundary conditions of the problem.

# Fields
- `ice_thickness`: an instance of [`AbstractIceThickness`](@ref) that defines how the ice thickness is updated.
- `viscous_displacement`: a boundary condition for the viscous displacement, defined as an [`OffsetBC`](@ref).
- `elastic_displacement`: a boundary condition for the elastic displacement, defined as an [`OffsetBC`](@ref).
- `sea_surface_perturbation`: a boundary condition for the sea surface perturbation, defined as an [`OffsetBC`](@ref).
"""
struct BoundaryConditions{
    T,      # <:AbstractFloat,
    M,      # <:AbstractMatrix{T},
    IT,     # <:AbstractIceThickness,
}
    ice_thickness::IT
    viscous_displacement::OffsetBC{T,M}
    elastic_displacement::OffsetBC{T,M}
    sea_surface_perturbation::OffsetBC{T,M}
end

function BoundaryConditions(
    domain::RegionalDomain{T,L,M};
    ice_thickness = ExternallyUpdatedIceThickness(),
    viscous_displacement = CornerBC(RegularBCSpace(), T(0)),
    elastic_displacement = CornerBC(ExtendedBCSpace(), T(0)),
    sea_surface_perturbation = CornerBC(ExtendedBCSpace(), T(0)),
) where {T<:AbstractFloat,L,M}

    # viscous_displacement must be defined on a regular grid
    @assert isa(viscous_displacement.space, RegularBCSpace)

    return BoundaryConditions(
        ice_thickness,
        precompute_bc(viscous_displacement, viscous_displacement.space, domain),
        precompute_bc(elastic_displacement, elastic_displacement.space, domain),
        precompute_bc(sea_surface_perturbation, sea_surface_perturbation.space, domain),
    )
end