mutable struct ColumnAnomalies{M}
    ice::M
    seawater::M
    sediment::M
    litho::M
    mantle::M
    load::M
    full::M
end

function ColumnAnomalies(domain)
    zero_columnanoms =
        [kernelzeros(domain) for _ in eachindex(fieldnames(ColumnAnomalies))]
    return ColumnAnomalies(zero_columnanoms...)
end

"""
$(TYPEDSIGNATURES)

Two-time-level buffers of the kinematic barystatic-sea-level formalism of
[adhikari_kinematic_2020](@citet), held by [`CurrentState`](@ref).

The formalism is intrinsically incremental: it needs the ice thickness, the
(signed) height above floatation and the land mask at the **start** of the
coupling interval, which no other part of the state keeps around (`sim.ref` is
the fixed reference, `sim.now` the current time). It writes its two per-cell
increments here as well, so the whole thing is one allocation-free broadcast.

# Fields
 - `H_ice_prev`: `H(t)`, ice thickness at the start of the interval (m).
 - `H_F_prev`: `H_F(t)`, height above floatation at the start of the interval (m).
 - `maskland_prev`: `ℒ(t)`, land mask at the start of the interval.
 - `delta_H_M`: `ΔH_M`, the component changing ocean **mass and volume** (their Eq. 11).
 - `delta_H_V`: `ΔH_V`, the component changing ocean **volume only** (their Eq. 12).

All five arrays are zero-size unless the simulation runs an
[`AdhikariBSLFormalism`](@ref), so [`GoelzerBSLFormalism`](@ref) pays nothing for them —
the same trick `u_K` uses for the Kelvin branches.
"""
struct KinematicBSL{M}
    H_ice_prev::M
    H_F_prev::M
    maskland_prev::M
    delta_H_M::M
    delta_H_V::M
end

function KinematicBSL(domain::RegionalDomain, active::Bool)
    T = eltype(domain.x)
    nx, ny = active ? (domain.nx, domain.ny) : (0, 0)
    return KinematicBSL(
        ntuple(_ -> kernelzeros(domain.backend, T, nx, ny), 5)...,
    )
end

# The zero-size arrays of an inactive `KinematicBSL` must not be written to with
# grid-sized data, so every writer that runs unconditionally checks this first.
kinematic_active(k::KinematicBSL) = length(k.H_ice_prev) > 0

abstract type AbstractState end

"""
$(TYPEDSIGNATURES)

Return a struct containing the reference state.
"""
struct ReferenceState{T,M,B} <: AbstractState

    u::M                    # viscous displacement
    ue::M                   # elastic displacement
    H_ice::M                # ref height of ice column
    H_af::M                 # ref height of ice column above floatation
    H_F::M                  # ref signed height above floatation (Adhikari eq. 8)
    H_water::M              # ref height of water column
    z_b::M                  # ref bedrock position
    z_ss::M                 # ref z_ss field
    V_af::T                 # ref sl-equivalent of ice volume above floatation
    V_pov::T                # ref potential ocean volume
    V_den::T                # ref potential ocean volume associated with V_den
    maskgrounded::B         # mask for grounded ice
    maskocean::B            # mask for ocean
end

# `maskgrounded`/`maskocean` are crisp `Bool` under `SharpTransition` but a
# continuous [0,1] field under `SmoothTransition`, so report an area fraction
# rather than a cell count — meaningful either way.
percent_string(mask) = string(round(100 * sum(mask) / length(mask), digits = 1), "%")

function Base.show(io::IO, ::MIME"text/plain", ref::ReferenceState)
    descriptors = [
        "V_af, V_pov, V_den" => [ref.V_af, ref.V_pov, ref.V_den],
        "extrema(u)" => extrema(ref.u),
        "extrema(ue)" => extrema(ref.ue),
        "extrema(H_ice)" => extrema(ref.H_ice),
        "extrema(z_b)" => extrema(ref.z_b),
        "extrema(z_ss)" => extrema(ref.z_ss),
        "grounded area" => percent_string(ref.maskgrounded),
        "ocean area" => percent_string(ref.maskocean),
    ]
    show_descriptors(io, descriptors)
end

"""
$(TYPEDSIGNATURES)

Return a mutable struct containing the geostate which will be updated over the simulation.
The geostate contains all the states of the [`Simulation`] to be solved.
"""
mutable struct CurrentState{T,M,K,B} <: AbstractState

    u::M                        # viscous displacement (total, = u_M + sum_j u_K[j])
    u_K::K                      # transient Kelvin-branch displacements at t_K, (nx, ny, N)
    u_K_next::K                 # same, pending for the end of the current step
    t_K::T                      # time at which u_K is valid
    ue::M                       # elastic displacement
    u_x::M                      # horizontal displacement in x
    u_y::M                      # horizontal displacement in y
    dudt::M                     # viscous displacement rate
    u_eq::M                     # equilibrium dispalcement
    H_ice::M                    # current height of ice column
    H_af::M                     # current height of ice column above floatation
    H_F::M                      # current signed height above floatation (Adhikari eq. 8)
    H_water::M                  # current height of water column
    columnanoms::ColumnAnomalies{M}             # column anomalies
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
    kinematic::KinematicBSL{M}  # two-time-level buffers of AdhikariBSLFormalism
    count_sparse_updates::Int   # count the updates that are sparser in time
end

# Initialise CurrentState from ReferenceState. `u_K` is a 3D array rather than a
# vector of matrices so that snapshot/restore, GPU transfer and AD treat it like
# any other field; see `KinematicBSL` above for the zero-size-when-unused trick.
function CurrentState(
    domain::RegionalDomain,
    ref::ReferenceState,
    z_bsl,
    nbranch::Int = 0,
    kinematic::Bool = false,
)
    T = eltype(domain.x)
    u_K = kernelzeros(domain.backend, T, domain.nx, domain.ny, nbranch)
    return CurrentState(
        copy(ref.u),                # u
        u_K,                        # u_K
        copy(u_K),                  # u_K_next
        T(0),                       # t_K  (overwritten by init_problem!/reset_state!)
        copy(ref.ue),               # ue
        kernelzeros(domain),         # u_x
        kernelzeros(domain),         # u_y
        kernelzeros(domain),         # dudt
        copy(ref.u),                # u_eq
        copy(ref.H_ice),            # H_ice
        copy(ref.H_af),             # H_af
        copy(ref.H_F),              # H_F
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
        KinematicBSL(domain, kinematic),    # kinematic
        0,                          # count_sparse_updates
    )
end

"""
$(TYPEDSIGNATURES)

(Re)initialise the two-time-level buffers of [`AdhikariBSLFormalism`](@ref) from the
reference state, so that the first coupling interval is well defined. A no-op
when the buffers are inactive, i.e. under [`GoelzerBSLFormalism`](@ref).
"""
function reset_kinematic!(k::KinematicBSL, ref::ReferenceState)
    kinematic_active(k) || return nothing
    k.H_ice_prev .= ref.H_ice
    k.H_F_prev .= ref.H_F
    k.maskland_prev .= not.(ref.maskocean)
    k.delta_H_M .= 0
    k.delta_H_V .= 0
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", now::CurrentState)
    descriptors = [
        "t_K" => now.t_K,
        "V_af, V_pov, V_den" => [now.V_af, now.V_pov, now.V_den],
        "delta_V" => now.delta_V,
        "z_bsl" => now.z_bsl,
        "extrema(u)" => extrema(now.u),
        "extrema(ue)" => extrema(now.ue),
        "extrema(H_ice)" => extrema(now.H_ice),
        "extrema(z_b)" => extrema(now.z_b),
        "extrema(z_ss)" => extrema(now.z_ss),
        "size(u_K)" => size(now.u_K),
        "grounded area" => percent_string(now.maskgrounded),
        "ocean area" => percent_string(now.maskocean),
        "count_sparse_updates" => now.count_sparse_updates,
    ]
    show_descriptors(io, descriptors)
end

"""
$(TYPEDSIGNATURES)

Reset the integrated + diagnostic fields of `sim.now` back to the initial
condition defined by `sim.ref`, and rewind the timer to `t_span[1]`. Used to
re-run the forward model from scratch (e.g. between inversion `loss`
evaluations) without reallocating the state. Does **not** touch model
parameters (viscosity, densities, ice snapshots), so it composes with
`reconstruct!`.
"""
function reset_state!(sim)
    now, ref = sim.now, sim.ref
    T = eltype(now.u)
    now.u .= ref.u
    now.u_K .= 0
    now.u_K_next .= 0
    now.t_K = T(sim.timer.t_span[1])
    now.ue .= ref.ue
    now.u_x .= 0
    now.u_y .= 0
    now.dudt .= 0
    now.u_eq .= ref.u
    now.H_ice .= ref.H_ice
    now.H_af .= ref.H_af
    now.H_F .= ref.H_F
    now.H_water .= ref.H_water
    for f in fieldnames(ColumnAnomalies)
        getfield(now.columnanoms, f) .= 0
    end
    now.z_b .= ref.z_b
    now.dz_ss .= 0
    now.z_ss .= ref.z_ss
    now.V_af = ref.V_af
    now.V_pov = ref.V_pov
    now.V_den = ref.V_den
    now.delta_V = T(0)
    now.z_bsl = T(sim.sealevel.bsl.z)
    now.maskgrounded .= ref.maskgrounded
    now.maskocean .= ref.maskocean
    reset_kinematic!(now.kinematic, ref)
    now.count_sparse_updates = 0
    sim.timer.t = sim.timer.t_span[1]
    empty!(sim.timer.t_computation)
    empty!(sim.timer.t_vec)
    return nothing
end