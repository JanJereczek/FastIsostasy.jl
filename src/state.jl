"""
$(TYPEDSIGNATURES)

Anomalies of the vertical columns, relative to the `ReferenceState`, that
load the solid Earth. Each is a density times a thickness anomaly.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ColumnAnomalies{M}
    "Ice column anomaly (kg m⁻²)"
    ice::M
    "Seawater column anomaly (kg m⁻²)"
    seawater::M
    "Sediment column anomaly (kg m⁻²)"
    sediment::M
    "Lithospheric column anomaly from the elastic displacement (kg m⁻²)"
    litho::M
    "Mantle column anomaly from the viscous displacement (kg m⁻²)"
    mantle::M
    "Surface load anomaly: ice + seawater + sediment, within the active mask (kg m⁻²)"
    load::M
    "Full anomaly: load + lithosphere + mantle, within the active mask (kg m⁻²)"
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
$(TYPEDFIELDS)

All five arrays are zero-size unless the simulation runs an
[`AdhikariBSLFormalism`](@ref), so [`GoelzerBSLFormalism`](@ref) pays nothing for them —
the same trick `u_K` uses for the Kelvin branches.
"""
struct KinematicBSL{M}
    "`H(t)`, ice thickness at the start of the interval (m)"
    H_ice_prev::M
    "`H_F(t)`, height above floatation at the start of the interval (m)"
    H_F_prev::M
    "`ℒ(t)`, land mask at the start of the interval"
    maskland_prev::M
    "`ΔH_M`, the component changing ocean **mass and volume** (their Eq. 11)"
    delta_H_M::M
    "`ΔH_V`, the component changing ocean **volume only** (their Eq. 12)"
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

# Fields
$(TYPEDFIELDS)
"""
struct ReferenceState{T,M,B} <: AbstractState
    "Viscous displacement (m)"
    u::M
    "Elastic displacement (m)"
    ue::M
    "Ice thickness (m)"
    H_ice::M
    "Ice thickness above flotation (m)"
    H_af::M
    "Signed height above flotation (m, Adhikari eq. 8)"
    H_F::M
    "Seawater thickness (m)"
    H_water::M
    "Bedrock elevation (m)"
    z_b::M
    "Sea-surface elevation (m)"
    z_ss::M
    "SLE ice volume above floatation (m^3)"
    V_af::T
    "SLE potential ocean volume (m^3)"
    V_pov::T
    "SLE volume associated with density difference (m^3)"
    V_den::T
    "Grounded ice mask (Bool)"
    maskgrounded::B
    "Ocean mask (Bool)"
    maskocean::B
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

# Fields
$(TYPEDFIELDS)

Total viscous displacement `u` is the sum of the transient Kelvin-branch displacements `u_K` and the viscous displacement `u_M` from the Maxwell branch, which is not stored separately.
"""
mutable struct CurrentState{T,M,K,B} <: AbstractState
    "Viscous displacement (m)"
    u::M
    "Kelvin-branch displacements at t_K, (nx, ny, N)"
    u_K::K
    "Kelvin-branch displacements at t_K + Δt, (nx, ny, N)"
    u_K_next::K
    "Time at which u_K is valid"
    t_K::T
    "Elastic displacement (m)"
    ue::M
    "Horizontal displacement in x (m)"
    u_x::M
    "Horizontal displacement in y (m)"
    u_y::M
    "Viscous displacement rate"
    dudt::M
    "Equilibrium viscous displacement (m)"
    u_eq::M
    "Ice thickness (m)"
    H_ice::M
    "Ice thickness above flotation (m)"
    H_af::M
    "Signed ice thickness above flotation (m)"
    H_F::M
    "Seawater thickness (m)"
    H_water::M
    "Column anomalies (kg m⁻²)"
    columnanoms::ColumnAnomalies{M}
    "Bedrock elevation (m)"
    z_b::M
    "SSH perturbation (m)"
    dz_ss::M
    "SSH (m)"
    z_ss::M
    "SLE ice volume above flotation (m^3)"
    V_af::T
    "SLE potential ocean volume (m^3)"
    V_pov::T
    "SLE volume associated with density difference (m^3)"
    V_den::T
    "Change in ocean volume over the last sparse update (m^3)"
    delta_V::T
    "Barystatic sea level (m)"
    z_bsl::T
    "Grounded ice mask (Bool)"
    maskgrounded::B
    "Ocean mask (Bool)"
    maskocean::B
    "Two-time-level buffers of the [`AdhikariBSLFormalism`](@ref)"
    kinematic::KinematicBSL{M}
    "Number of sparse diagnostic updates performed so far"
    count_sparse_updates::Int
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
    sim.timer.t_sparse0 = sim.timer.t_span[1]
    empty!(sim.timer.t_computation)
    empty!(sim.timer.t_vec)
    return nothing
end