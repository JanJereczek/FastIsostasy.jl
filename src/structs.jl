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
    Theta::AbstractMatrix{T}
    Lat::AbstractMatrix{T}
    Lon::AbstractMatrix{T}
    K::AbstractMatrix{T}
    pseudodiff::AbstractMatrix{T}
    harmonic::AbstractMatrix{T}
    biharmonic::AbstractMatrix{T}
    use_cuda::Bool
    arraykernel     # Array or CuArray depending on chosen hardware
end

struct PhysicalConstants{T<:AbstractFloat}
    g::T
    seconds_per_year::T
    ice_density::T
    seawater_density::T
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

mutable struct GeoState{T<:AbstractFloat}
    hi::Matrix{T}               # current height of ice column
    hi_ref::Matrix{T}           # reference
    hw::Matrix{T}               # current height of water column
    hw_ref::Matrix{T}           # reference
    b::Matrix{T}                # vertical bedrock position
    b_ref::Matrix{T}            # reference
    geoid::Matrix{T}            # current geoid displacement
    sealevel::Matrix{T}         # current sealevel field
    sealevel_ref::Matrix{T}     # reference sealevel field
    volume_pov::T               # current potential ocean volume
    volume_pov_ref::T           # reference
    slc_pov::T                  # sea-level contribution associated with volume_pov
    volume_den::T               # potential ocean volume associated with density differences
    volume_den_ref::T           # reference
    slc_den::T                  # sea-level contribution associated with volume_den
    slc::T                      # total sealevel contribution
    conservation_term::T        # a term for mass conservation
end

struct PrecomputedFastiso{T<:AbstractFloat}
    elasticgreen::AbstractMatrix{T}
    fourier_elasticgreen::AbstractMatrix{Complex{T}}
    pfft::AbstractFFTs.Plan
    pifft::AbstractFFTs.ScaledPlan
    Dx::AbstractMatrix{T}
    Dy::AbstractMatrix{T}
    Dxx::AbstractMatrix{T}
    Dyy::AbstractMatrix{T}
    Dxy::AbstractMatrix{T}
    negligible_gradD::Bool
    rhog::T
    geoidgreen::AbstractMatrix{T}
end

struct ODEParams{T<:AbstractFloat}
    Omega::ComputationDomain{T}
    c::PhysicalConstants{T}
    p::MultilayerEarth{T}
    Hice::Interpolations.Extrapolation
    tools::PrecomputedFastiso{T}
end

"""

    FastIsoResults

A mutable struct containing the results of FastIsostasy:
    - `t_out` the time output vector
    - `u3D_elastic` the elastic response over `t_out`
    - `u3D_viscous` the viscous response over `t_out`
    - `dudt3D_viscous` the displacement rate over `t_out`
    - `geoid3D` the geoid response over `t_out`
    - `Hice` an interpolator of the ice thickness over time
    - `eta` an interpolator of the upper-mantle viscosity over time
"""
struct FastIsoResults{T<:AbstractFloat}
    t_out::Vector{T}
    viscous::Vector{Matrix{T}}
    displacement_rate::Vector{Matrix{T}}
    elastic::Vector{Matrix{T}}
    geoid::Vector{Matrix{T}}
    sealevel::Vector{Matrix{T}}
    Hice::Interpolations.Extrapolation
    eta::Interpolations.Extrapolation
end