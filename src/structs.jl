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
    H_ice::Matrix{T}               # current height of ice column
    H_ice_ref::Matrix{T}           # reference
    H_water::Matrix{T}               # current height of water column
    H_water_ref::Matrix{T}           # reference
    b::Matrix{T}                # vertical bedrock position
    b_ref::Matrix{T}            # reference
    geoid::Matrix{T}            # current geoid displacement
    z0::Matrix{T}               # reference height to allow external sea-level forcing
    sealevel::Matrix{T}         # current sealevel field
    sealevel_ref::Matrix{T}     # reference sealevel field
    V_af::T                     # ice volume above floatation
    sle_af::T                   # sl-equivalent of ice volume above floatation
    sle_af_ref::T               # reference sl-equivalent of ice volume above floatation
    slc_af::T                   # sl-contribution of ice volume above floatation
    V_pov::T                    # current potential ocean volume
    V_pov_ref::T                # reference
    slc_pov::T                  # sea-level contribution associated with V_pov
    V_den::T                    # potential ocean volume associated with density differences
    V_den_ref::T                # reference
    slc_den::T                  # sea-level contribution associated with V_den
    slc::T                      # total sealevel contribution
    conservation_term::T        # a term for mass conservation
end

struct ReferenceGeoState{T<:AbstractFloat}
    H_ice::Matrix{T}           # reference
    H_water::Matrix{T}           # reference
    b::Matrix{T}            # reference
    z0::Matrix{T}               # reference height to allow external sea-level forcing
    sealevel::Matrix{T}     # reference sealevel field
    V_af::T                 # ice volume above floatation
    sle_af::T               # reference sl-equivalent of ice volume above floatation
    V_pov::T                # reference
    V_den::T                # reference
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

struct SuperStruct{T<:AbstractFloat}
    Omega::ComputationDomain{T}
    c::PhysicalConstants{T}
    p::MultilayerEarth{T}
    Hice::Interpolations.Extrapolation
    Hice_cpu::Interpolations.Extrapolation
    tools::PrecomputedFastiso{T}
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