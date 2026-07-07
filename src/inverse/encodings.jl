# =============================================================================
# Encodings.
#
# An encoding maps a low-dimensional parameter vector `θ` onto the
# high-dimensional model inputs (effective viscosity, densities, ice-thickness
# snapshots) via `reconstruct!(sim, θ, encoding)`. It must be Enzyme-legal: plain
# indexed reads of `θ` and in-place broadcasts into `sim` fields, no closures over
# untracked state, no scalar mutation that depends on `θ` in a non-differentiable
# way.
#
# `nparams(encoding)` returns `length(θ)`.
#
# Concrete encodings implemented here: `Test1Encoding` (joint Vialov ice + bimodal
# viscosity, for Test 1) and `Test2Encoding` (4-Gaussian viscosity + densities,
# for Test 2). `EOFEncoding` / `AutoEncoding` / `VariationalAutoEncoding` are
# stubs — their decoders are trained outside FastIsostasy; only the decoder
# application needs to live here and be Enzyme-legal.
# =============================================================================

abstract type AbstractEncoding{T<:AbstractFloat} end

function nparams end

# --- Enzyme-legal field builders --------------------------------------------

# Add an isotropic Gaussian bump `amp * exp(-r²/2σ²)` centred at (μx, μy) to
# `field`. Differentiable w.r.t. μx, μy, σ, amp.
function add_gaussian!(field, X, Y, μx, μy, σ, amp)
    @. field += amp * exp(-((X - μx)^2 + (Y - μy)^2) / (2 * σ^2))
    return nothing
end

# Add a radially-symmetric Vialov dome of central thickness `Hc`, radius `L`,
# centred at (xc, yc): H(r) = Hc * max(1 − (r/L)^(4/3), 0)^(3/8).
# NB: the (·)^(3/8) has an infinite slope at the margin (base → 0⁺); gradients
# w.r.t. the centre are steep there but the bulk dominates.
function add_vialov!(H, X, Y, xc, yc, L, Hc)
    @. H += Hc * max(1 - (sqrt((X - xc)^2 + (Y - yc)^2) / L)^(4//3), 0)^(3//8)
    return nothing
end

# Write the physical viscosity field from a log10 field held in `logη`.
set_viscosity_from_log10!(sim, logη) = (@. sim.solidearth.effective_viscosity = 10^logη; nothing)

# The ice-thickness snapshots an encoding writes into (for time interpolation).
ice_snapshots(sim) = sim.bcs.ice_thickness.H_itp.X

# =============================================================================
# Test1Encoding — joint Vialov ice (3 domes, time-varying) + bimodal viscosity.
# =============================================================================

"""
    Test1Encoding(knot_times, radii, visc_amps)

Encoding for Test 1. Fixed config: `knot_times` (K ice-interpolation times),
`radii` (the 3 fixed Vialov radii `Lᵢ`), `visc_amps` (the 2 fixed log10 anomaly
amplitudes, e.g. `(-1, +1)` decades).

θ layout (length `3K + 13`):
`[Hc₁(1:K), Hc₂(1:K), Hc₃(1:K), x₁,y₁, x₂,y₂, x₃,y₃, log10η_bg, μ₁ₓ,μ₁ᵧ,σ₁, μ₂ₓ,μ₂ᵧ,σ₂]`.
"""
struct Test1Encoding{T} <: AbstractEncoding{T}
    knot_times::Vector{T}
    radii::NTuple{3, T}
    visc_amps::NTuple{2, T}
end

nparams(enc::Test1Encoding) = 3 * length(enc.knot_times) + 13

function reconstruct!(sim, θ, enc::Test1Encoding)
    X, Y = sim.domain.X, sim.domain.Y
    K = length(enc.knot_times)

    # --- viscosity: background + two log10 Gaussian anomalies ---
    off = 3K + 6
    logη = fill!(similar(sim.solidearth.effective_viscosity), θ[off + 1])
    add_gaussian!(logη, X, Y, θ[off + 2], θ[off + 3], θ[off + 4], enc.visc_amps[1])
    add_gaussian!(logη, X, Y, θ[off + 5], θ[off + 6], θ[off + 7], enc.visc_amps[2])
    set_viscosity_from_log10!(sim, logη)

    # --- ice: 3 Vialov domes summed at each knot time ---
    snaps = ice_snapshots(sim)
    for k in 1:K
        Hk = snaps[k]
        fill!(Hk, zero(eltype(Hk)))
        for i in 1:3
            xc = θ[3K + 2i - 1]
            yc = θ[3K + 2i]
            Hc = θ[(i - 1) * K + k]
            add_vialov!(Hk, X, Y, xc, yc, enc.radii[i], Hc)
        end
    end
    return nothing
end

# =============================================================================
# Test2Encoding — 4-Gaussian viscosity + densities (ice known/fixed).
# =============================================================================

"""
    Test2Encoding()

Encoding for Test 2. θ layout (length 19):
`[log10η_bg, (μₓ,μᵧ,σ,amp)×4, ρ_uppermantle, ρ_litho]`. Ice thickness is not
touched (assumed known).
"""
struct Test2Encoding{T} <: AbstractEncoding{T} end
Test2Encoding(; T = Float32) = Test2Encoding{T}()

nparams(::Test2Encoding) = 19

function reconstruct!(sim, θ, ::Test2Encoding)
    X, Y = sim.domain.X, sim.domain.Y
    logη = fill!(similar(sim.solidearth.effective_viscosity), θ[1])
    for i in 1:4
        b = 1 + (i - 1) * 4
        add_gaussian!(logη, X, Y, θ[b + 1], θ[b + 2], θ[b + 3], θ[b + 4])
    end
    set_viscosity_from_log10!(sim, logη)
    sim.solidearth.rho_uppermantle = θ[18]
    sim.solidearth.rho_litho = θ[19]
    return nothing
end

# =============================================================================
# Trained-decoder encodings (stubs — decoders trained outside FastIsostasy).
# =============================================================================

"""
    EOFEncoding

Empirical-orthogonal-function (linear) decoder: `field = mean + Modes * θ`.
Stub — the decoder application must be implemented Enzyme-legally.
"""
struct EOFEncoding{T} <: AbstractEncoding{T} end

"""
    AutoEncoding

Autoencoder decoder. Stub.
"""
struct AutoEncoding{T} <: AbstractEncoding{T} end

"""
    VariationalAutoEncoding

Variational-autoencoder decoder. Stub.
"""
struct VariationalAutoEncoding{T} <: AbstractEncoding{T} end

for E in (:EOFEncoding, :AutoEncoding, :VariationalAutoEncoding)
    @eval nparams(::$E) = error(string($E) * " is not implemented yet.")
    @eval reconstruct!(sim, θ, ::$E) = error(string($E) * " is not implemented yet.")
end
