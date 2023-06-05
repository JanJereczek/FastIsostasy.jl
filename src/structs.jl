XMatrix = Union{Matrix{T}, CuArray{T, 2}} where {T<:Real}

"""
    ComputationDomain

Return a struct containing all information related to geometry of the domain
and potentially used parallelism.
"""
struct ComputationDomain{T<:AbstractFloat}
    Wx::T                       # Domain length in x (m)
    Wy::T                       # Domain length in y (m)
    N::Int                      # Average number of grid points along one dimension
    N2::Int                     # N/2
    dx::T                       # Spatial discretization in x
    dy::T                       # Spatial discretization in y
    x::Vector{T}
    y::Vector{T}
    X::XMatrix
    Y::XMatrix
    R::XMatrix
    Theta::XMatrix
    Lat::XMatrix
    Lon::XMatrix
    K::XMatrix
    null::XMatrix         # a zero matrix of size Nx x Ny
    pseudodiff::XMatrix   # pseudodiff operator
    harmonic::XMatrix     # harmonic operator
    biharmonic::XMatrix   # biharmonic operator
    use_cuda::Bool
    arraykernel                     # Array or CuArray depending on chosen hardware
end

"""
    PhysicalConstants

Return a struct containing important physical constants.
"""
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

"""
    MultilayerEarth

Return a struct containing all information related to the radially layered structure of the solid Earth and
its parameters.
"""
mutable struct MultilayerEarth{T<:AbstractFloat}
    mean_gravity::T
    mean_density::T
    effective_viscosity::XMatrix
    litho_thickness::XMatrix
    litho_rigidity::XMatrix
    litho_poissonratio::T
    layers_density::Vector{T}
    layer_viscosities::Array{T, 3}
    layer_boundaries::Array{T, 3}
end

"""
    RefSealevelState

Return a struct containing the reference slstate. We define the slstate to be all quantities related to sea-level.
"""
struct RefSealevelState{T<:AbstractFloat}
    H_ice::XMatrix        # reference height of ice column
    H_water::XMatrix      # reference height of water column
    b::XMatrix            # reference bedrock position
    z0::XMatrix           # reference height to allow external sea-level forcing
    sealevel::XMatrix     # reference sealevel field
    sle_af::T                       # reference sl-equivalent of ice volume above floatation
    V_pov::T                        # reference potential ocean volume
    V_den::T                        # reference potential ocean volume associated with V_den
    conservation_term::T            # a term for mass conservation
end

"""
    SealevelState

Return a mutable struct containing the slstate which will be updated over the simulation.
"""
mutable struct SealevelState{T<:AbstractFloat}
    H_ice::XMatrix          # current height of ice column
    H_water::XMatrix        # current height of water column
    b::XMatrix              # vertical bedrock position
    geoid::XMatrix          # current geoid displacement
    sealevel::XMatrix       # current sealevel field
    V_af::T                 # ice volume above floatation
    sle_af::T               # sl-equivalent of ice volume above floatation
    slc_af::T               # sl-contribution of Vice above floatation
    V_pov::T                # current potential ocean volume
    slc_pov::T              # sea-level contribution associated with V_pov
    V_den::T                # potential ocean volume associated with density differences
    slc_den::T              # sea-level contribution associated with V_den
    slc::T                  # total sealevel contribution
    countupdates::Int       # count the updates of the slstate
    dt::T                   # update step
end

"""
    PrecomputedFastiso

Return a struct containing all the pre-computed terms needed for the forward integration of the model.
"""
struct PrecomputedFastiso{T<:AbstractFloat}
    elasticgreen::XMatrix
    fourier_elasticgreen::XMatrix{Complex{T}}
    pfft::AbstractFFTs.Plan
    pifft::AbstractFFTs.ScaledPlan
    Dx::XMatrix
    Dy::XMatrix
    Dxx::XMatrix
    Dyy::XMatrix
    Dxy::XMatrix
    negligible_gradD::Bool
    rhog::T
    geoidgreen::XMatrix
end

"""
    SuperStruct

Return a struct containing all the other structs needed for the forward integration of the model.
"""
struct SuperStruct{T<:AbstractFloat}
    Omega::ComputationDomain{T}
    c::PhysicalConstants{T}
    p::MultilayerEarth{T}
    tools::PrecomputedFastiso{T}
    Hice::Interpolations.Extrapolation
    Hice_cpu::Interpolations.Extrapolation
    eta::Interpolations.Extrapolation
    eta_cpu::Interpolations.Extrapolation
    refslstate::RefSealevelState{T}
    slstate::SealevelState{T}
    interactive_sealevel::Bool
end

"""

    FastisoResults

A struct containing the results of FastIsostasy:
    - `t_out` the time output vector
    - `u3D_elastic` the elastic response over `t_out`
    - `u3D_viscous` the viscous response over `t_out`
    - `dudt3D_viscous` the displacement rate over `t_out`
    - `geoid3D` the geoid response over `t_out`
    - `Hice` an interpolator of the ice thickness over time
    - `eta` an interpolator of the upper-mantle viscosity over time
"""
struct FastisoResults{T<:AbstractFloat}
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