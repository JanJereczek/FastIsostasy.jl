"""
    AbstractBC

An abstract type representing a boundary condition in the context of a computational domain.
Available subtypes are:
- [`OffsetBC`](@ref)
"""
abstract type AbstractBC end

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
    OffsetBC{T} <: AbstractBC

A boundary condition that applies an offset to the values at the boundaries of a
computational domain. Contains:
- `space`: the [`AbstractBCSpace`](@ref) in which the boundary condition is defined.
- `x_border`: the offset value to be applied at the boundaries.
- `W`: a weight matrix to apply the boundary condition according to some [ÀbstractBCRule](@ref).
"""
struct OffsetBC{T} <: AbstractBC
    space::AbstractBCSpace
    x_border::T
    W::KernelMatrix{T}
end

"""
    apply_bc!(X, bc::OffsetBC)

Apply the boundary condition `bc` to the matrix `X` in-place.
"""
function apply_bc!(X, bc::OffsetBC)
    @. X = muladd(X + bc.x_border, -bc.W, X)
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
    precompute_bc(bc::AbstractBC, sp::AbstractBCSpace, Omega::ComputationDomain)

Precompute the boundary condition for the given computation domain.
"""
function precompute_bc(bc::CornerBC, sp::RegularBCSpace, Omega::ComputationDomain)
    W = corner_ones(eltype(Omega.R), Omega.nx, Omega.ny)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

function precompute_bc(bc::CornerBC, sp::ExtendedBCSpace, Omega::ComputationDomain)
    W = corner_ones(eltype(Omega.R), 2*Omega.nx-1, 2*Omega.ny-1)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

function precompute_bc(bc::BorderBC, sp::RegularBCSpace, Omega::ComputationDomain)
    W = border_ones(eltype(Omega.R), Omega.nx, Omega.ny)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

function precompute_bc(bc::BorderBC, sp::ExtendedBCSpace, Omega::ComputationDomain)
    W = border_ones(eltype(Omega.R), 2*Omega.nx-1, 2*Omega.ny-1)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

function precompute_bc(bc::DistanceWeightedBC, sp::RegularBCSpace, Omega::ComputationDomain)
    W = border_ones(eltype(Omega.R), Omega.nx, Omega.ny)
    W .= W .* Omega.R
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

function precompute_bc(bc::DistanceWeightedBC, sp::ExtendedBCSpace, Omega::ComputationDomain)
    error("DistanceWeightedBC is not implemented for ExtendedBCSpace")
end

function precompute_bc(bc::MeanBC, sp::RegularBCSpace, Omega::ComputationDomain)
    W = ones(eltype(Omega.R), Omega.nx, Omega.ny)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

function precompute_bc(bc::MeanBC, sp::ExtendedBCSpace, Omega::ComputationDomain)
    W = ones(eltype(Omega.R), 2*Omega.nx-1, 2*Omega.ny-1)
    norm!(W)
    return OffsetBC(bc.space, bc.x_border, Omega.arraykernel(W))
end

#########################################################################
# FastIsoProblem level
#########################################################################

"""
    ProblemBCs(viscous_displacement, elastic_displacement, geoid_perturbation, Omega)

Create a `ProblemBCs` struct containing the boundary conditions for the problem.
"""
struct ProblemBCs{T}
    u::OffsetBC{T}           # viscous displacement
    u_e::OffsetBC{T}         # elastic displacement
    dz_ss::OffsetBC{T}       # sea surface height
end

function ProblemBCs(
    Omega::ComputationDomain{T, L, M};
    viscous_displacement = CornerBC(RegularBCSpace(), T(0)),
    elastic_displacement = CornerBC(ExtendedBCSpace(), T(0)),
    geoid_perturbation = CornerBC(ExtendedBCSpace(), T(0))) where
    {T<:AbstractFloat, L, M}
    
    # viscous_displacement must be defined on a regular grid
    @assert isa(viscous_displacement.space, RegularBCSpace)
    
    return ProblemBCs(
        precompute_bc(viscous_displacement, viscous_displacement.space, Omega),
        precompute_bc(elastic_displacement, elastic_displacement.space, Omega),
        precompute_bc(geoid_perturbation, geoid_perturbation.space, Omega)
    )
end