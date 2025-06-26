#########################################################
# Geostate
#########################################################
abstract type AbstractState end

"""
    ReferenceState

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
    maskactive::B
end

mutable struct ColumnAnomalies{T<:AbstractFloat, M<:KernelMatrix{T}}
    load::M
    litho::M
    mantle::M
    full::M
end

function ColumnAnomalies(Omega)
    zero_columnanoms = [kernelnull(Omega) for _ in eachindex(fieldnames(ColumnAnomalies))]
    return ColumnAnomalies(zero_columnanoms...)
end

"""
    CurrentState

Return a mutable struct containing the geostate which will be updated over the simulation.
The geostate contains all the states of the [`FastIsoProblem`] to be solved.
"""
mutable struct CurrentState{
    T<:AbstractFloat,
    M<:KernelMatrix{T},
    B<:BoolMatrix,
    OS<:AbstractBSL,
} <: AbstractState

    u::M                    # viscous displacement
    ue::M                   # elastic displacement
    u_x::M                  # horizontal displacement in x
    u_y::M                  # horizontal displacement in y
    dudt::M                 # viscous displacement rate
    u_eq::M                 # equilibrium dispalcement
    H_ice::M                # current height of ice column
    H_af::M                 # current height of ice column above floatation
    H_water::M              # current height of water column
    columnanoms::ColumnAnomalies{T, M}         # column anomalies
    z_b::M                  # vertical bedrock position
    dz_ss::M                # current z_ss perturbation
    z_ss::M                 # current z_ss field
    V_af::T                 # V contribution from ice above floatation
    V_pov::T                # V contribution from bedrock adjustment
    V_den::T                # V contribution from diff between melt- and saltwater density
    maskgrounded::B         # mask for grounded ice
    maskocean::B            # mask for ocean
    bsl::OS                  # ocean surface change
    countupdates::Int       # count the updates of the geostate
    k::Int                  # index of the t_out segment
end

# Initialise CurrentState from ReferenceState
function CurrentState(Omega::RegionalComputationDomain, ref::ReferenceState, bsl)
    return CurrentState(
        copy(ref.u), copy(ref.ue), kernelnull(Omega), kernelnull(Omega),
        kernelnull(Omega), copy(ref.u),
        copy(ref.H_ice), copy(ref.H_af), copy(ref.H_water),
        ColumnAnomalies(Omega), copy(ref.z_b),
        kernelnull(Omega), copy(ref.z_ss),
        copy(ref.V_af), copy(ref.V_pov), copy(ref.V_den),
        copy(ref.maskgrounded), copy(ref.maskocean),
        bsl, 0, 1,
    )
end