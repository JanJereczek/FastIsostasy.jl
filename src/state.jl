mutable struct ColumnAnomalies{T<:AbstractFloat, M<:KernelMatrix{T}}
    ice::M
    seawater::M
    sediment::M
    litho::M
    mantle::M
    load::M
    full::M
end

function ColumnAnomalies(domain)
    zero_columnanoms = [kernelzeros(domain) for _ in eachindex(fieldnames(ColumnAnomalies))]
    return ColumnAnomalies(zero_columnanoms...)
end

abstract type AbstractState end

"""
$(TYPEDSIGNATURES)

Return a struct containing the reference state.
"""
struct ReferenceState{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    B<:BoolMatrix,
} <: AbstractState

    u::M                    # viscous displacement
    ue::M                   # elastic displacement
    H_ice::M                # ref height of ice column
    H_af::M                 # ref height of ice column above floatation
    H_water::M              # ref height of water column
    z_b::M                  # ref bedrock position
    z_ss::M                 # ref z_ss field
    V_af::T                 # ref sl-equivalent of ice volume above floatation
    V_pov::T                # ref potential ocean volume
    V_den::T                # ref potential ocean volume associated with V_den
    maskgrounded::B         # mask for grounded ice
    maskocean::B            # mask for ocean
end

"""
$(TYPEDSIGNATURES)

Return a mutable struct containing the geostate which will be updated over the simulation.
The geostate contains all the states of the [`Simulation`] to be solved.
"""
mutable struct CurrentState{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    B<:BoolMatrix,
} <: AbstractState

    u::M                        # viscous displacement
    ue::M                       # elastic displacement
    u_x::M                      # horizontal displacement in x
    u_y::M                      # horizontal displacement in y
    dudt::M                     # viscous displacement rate
    u_eq::M                     # equilibrium dispalcement
    H_ice::M                    # current height of ice column
    H_af::M                     # current height of ice column above floatation
    H_water::M                  # current height of water column
    columnanoms::ColumnAnomalies{T, M}         # column anomalies
    z_b::M                      # vertical bedrock position
    dz_ss::M                    # current z_ss perturbation
    z_ss::M                     # current z_ss field
    V_af::T                     # V contribution from ice above floatation
    V_pov::T                    # V contribution from bedrock adjustment
    V_den::T                    # V contribution from diff between melt- and saltwater density
    delta_V::T                  # change in volume
    z_bsl::T                    # ocean surface change
    maskgrounded::B             # mask for grounded ice
    maskocean::B                # mask for ocean
    count_sparse_updates::Int   # count the updates that are sparser in time
end

# Initialise CurrentState from ReferenceState
function CurrentState(domain::RegionalDomain, ref::ReferenceState, z_bsl)
    T = eltype(domain.x)
    return CurrentState(
        copy(ref.u),                # u
        copy(ref.ue),               # ue
        kernelzeros(domain),         # u_x
        kernelzeros(domain),         # u_y
        kernelzeros(domain),         # dudt
        copy(ref.u),                # u_eq
        copy(ref.H_ice),            # H_ice
        copy(ref.H_af),             # H_af
        copy(ref.H_water),          # H_water
        ColumnAnomalies(domain),    # columnanoms
        copy(ref.z_b),              # z_b
        kernelzeros(domain),         # dz_ss   (can init to 0 because diagnostic variable)
        copy(ref.z_ss),             # z_ss
        copy(ref.V_af),             # V_af
        copy(ref.V_pov),            # V_pov
        copy(ref.V_den),            # V_den
        T(0),                       # delta_V
        T(z_bsl),                   # z_bsl
        copy(ref.maskgrounded),     # maskgrounded
        copy(ref.maskocean),        # maskocean
        0,                          # count_sparse_updates
    )
end