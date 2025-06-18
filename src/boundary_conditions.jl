abstract type AbstractBC end

abstract type BCSpace end
struct RegularBCSpace <: BCSpace end
struct ExtendedBCSpace <: BCSpace end

#########################################################################
# Computation-level
#########################################################################

struct OffsetBC{T} <: AbstractBC
    space::BCSpace
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

struct CornerBC
    space::BCSpace
    x_border
end

struct BorderBC
    space::BCSpace
    x_border
end

struct DistanceWeightedBC
    space::BCSpace
    x_border
end

struct MeanBC
    space::BCSpace
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
    precompute_bc(bc::AbstractBC, sp::BCSpace, Omega::ComputationDomain)

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