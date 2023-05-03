struct ComputationDomain{T<:AbstractFloat}
    Lx::T                       # Domain length in x (m)
    Ly::T                       # Domain length in y (m)
    N::Int                      # Average number of grid points along one dimension
    N2::Int                     # N/2
    dx::T                       # Spatial discretization in x
    dy::T                       # Spatial discretization in y
    x::Vector{T}
    y::Vector{T}
    X::AbstractMatrix{T}
    Y::AbstractMatrix{T}
    R::AbstractMatrix{T}
    Theta::AbstractMatrix{T}
    Lat::AbstractMatrix{T}
    Lon::AbstractMatrix{T}
    K::AbstractMatrix{T}
    null::AbstractMatrix{T}         # a zero matrix of size Nx x Ny
    pseudodiff::AbstractMatrix{T}   # pseudodiff operator
    harmonic::AbstractMatrix{T}     # harmonic operator
    biharmonic::AbstractMatrix{T}   # biharmonic operator
    use_cuda::Bool
    arraykernel                     # Array or CuArray depending on chosen hardware
end

struct PhysicalConstants{T<:AbstractFloat}
    mE::T
    r_equator::T
    r_pole::T
    A_ocean::T
    g::T
    G::T
    seconds_per_year::T
    rho_ice::T
    rho_water::T
    rho_seawater::T
    rho_core::T
    rho_topastheno::T
end

mutable struct MultilayerEarth{T<:AbstractFloat}
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

struct ReferenceGeoState{T<:AbstractFloat}
    H_ice::AbstractMatrix{T}        # reference height of ice column
    H_water::AbstractMatrix{T}      # reference height of water column
    b::AbstractMatrix{T}            # reference bedrock position
    z0::AbstractMatrix{T}           # reference height to allow external sea-level forcing
    sealevel::AbstractMatrix{T}     # reference sealevel field
    sle_af::T                       # reference sl-equivalent of ice volume above floatation
    V_pov::T                        # reference potential ocean volume
    V_den::T                        # reference potential ocean volume associated with V_den
    conservation_term::T            # a term for mass conservation
end

mutable struct GeoState{T<:AbstractFloat}
    H_ice::AbstractMatrix{T}        # current height of ice column
    H_water::AbstractMatrix{T}      # current height of water column
    b::AbstractMatrix{T}            # vertical bedrock position
    geoid::AbstractMatrix{T}        # current geoid displacement
    sealevel::AbstractMatrix{T}     # current sealevel field
    V_af::T                         # ice volume above floatation
    sle_af::T                       # sl-equivalent of ice volume above floatation
    slc_af::T                       # sl-contribution of Vice above floatation
    V_pov::T                        # current potential ocean volume
    slc_pov::T                      # sea-level contribution associated with V_pov
    V_den::T                        # potential ocean volume associated with density differences
    slc_den::T                      # sea-level contribution associated with V_den
    slc::T                          # total sealevel contribution
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

struct SuperStruct{T<:AbstractFloat}
    Omega::ComputationDomain{T}
    c::PhysicalConstants{T}
    p::MultilayerEarth{T}
    Hice::Interpolations.Extrapolation
    Hice_cpu::Interpolations.Extrapolation
    tools::PrecomputedFastiso{T}
    refgeostate::ReferenceGeoState{T}
    geostate::GeoState{T}
    active_geostate::Bool
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
    tools::PrecomputedFastiso{T}
    viscous::Vector{Matrix{T}}
    displacement_rate::Vector{Matrix{T}}
    elastic::Vector{Matrix{T}}
    geoid::Vector{Matrix{T}}
    sealevel::Vector{Matrix{T}}
    Hice::Interpolations.Extrapolation
    eta::Interpolations.Extrapolation
end