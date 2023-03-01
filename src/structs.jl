
struct ComputationDomain{T<:AbstractFloat}
    Lx::T           # Domain length in x (m)
    Ly::T           # Domain length in y (m)
    N::Int          # Average number of grid points in one dimension (1)
    N2::Int         # N/2
    dx::T           # Spatial discretization in x
    dy::T           # Spatial discretization in y
    x::Vector{T}
    y::Vector{T}
    X::AbstractMatrix{T}
    Y::AbstractMatrix{T}
    R::AbstractMatrix{T}
    Θ::AbstractMatrix{T}
    loadresponse_matrix::AbstractMatrix{T}
    loadresponse_function::Function
    pseudodiff_coeffs::AbstractMatrix{T}
    harmonic_coeffs::AbstractMatrix{T}
    biharmonic_coeffs::AbstractMatrix{T}
    use_cuda::Bool
end

struct PhysicalConstants{T<:AbstractFloat}
    g::T
    seconds_per_year::T
    ice_density::T
    r_equator::T
    r_pole::T
    G::T
    mE::T
    rho_0::T
    rho_1::T
end

struct MultilayerEarth{T<:AbstractFloat}
    mean_gravity::T
    mean_density::T
    effective_viscosity::AbstractMatrix{T}
    litho_thickness::AbstractMatrix{T}
    litho_rigidity::AbstractMatrix{T}
    litho_poissonratio::T
    layers_density::Vector{T}
    layers_viscosity::Array{T, 3}
    layers_begin::Array{T, 3}
end

mutable struct ColumnChanges{T<:AbstractFloat}
    hi::AbstractMatrix{T}
    hi0::AbstractMatrix{T}
    hw::AbstractMatrix{T}
    hw0::AbstractMatrix{T}
    b::AbstractMatrix{T}
    b0::AbstractMatrix{T}
end

struct PrecomputedFastiso{T<:AbstractFloat}
    loadresponse::AbstractMatrix{T}
    fourier_loadresponse::AbstractMatrix{Complex{T}}
    pfft::AbstractFFTs.Plan
    pifft::AbstractFFTs.ScaledPlan
    Dx::AbstractMatrix{T}
    Dy::AbstractMatrix{T}
    Dxx::AbstractMatrix{T}
    Dyy::AbstractMatrix{T}
    Dxy::AbstractMatrix{T}
    negligible_gradD::Bool
    rhog::T
    geoid_green::AbstractMatrix{T}
end
