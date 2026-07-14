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

"""
    AbstractEncoding{T}

Supertype of the encodings, which map a low-dimensional parameter vector `θ` onto
the high-dimensional model inputs (effective viscosity, densities, ice-thickness
snapshots) through [`reconstruct!`](@ref). Encoding the unknowns is what makes
forward-mode ([`TangentMode`](@ref)) inversion affordable.

Concrete encodings: [`Test1Encoding`](@ref) (joint Vialov ice + bimodal
viscosity), [`Test2Encoding`](@ref) (4-Gaussian viscosity + densities).
[`EOFEncoding`](@ref) / [`AutoEncoding`](@ref) / [`VariationalAutoEncoding`](@ref)
are stubs for trained decoders.

Passing `encoding = nothing` to an inversion instead selects full-field control
(one unknown per grid cell); see [`ParameterInversion`](@ref).

An encoding must be Enzyme-legal: indexed reads of `θ` and in-place broadcasts
into `sim` fields, no closures over untracked state.
"""
abstract type AbstractEncoding{T<:AbstractFloat} end

"""
    nparams(encoding) -> Int

The number of parameters the encoding expects, i.e. the required `length(θ)`.
"""
function nparams end

# --- Enzyme-legal field builders --------------------------------------------

"""
$(TYPEDSIGNATURES)

Add an isotropic Gaussian bump `amp * exp(-r²/2σ²)` centred at `(μx, μy)` to `field`. Differentiable w.r.t. `μx`, `μy`, `σ`, `amp`.
"""
function add_gaussian!(field, X, Y, μx, μy, σ, amp)
    @. field += amp * exp(-((X - μx)^2 + (Y - μy)^2) / (2 * σ^2))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Vialov shape function `max(base, 0)^(3/8)`, guarded against the pow rule's
`Inf·0 = NaN` at the clamped margin: for `base ≤ 0`, `max(base,0) = 0` with a
*correct* zero forward-mode tangent (the constant branch of `max` is exactly
0, tangent 0), but differentiating `0^(3//8)` itself still forms `(3/8)·0^(-5/8)
= Inf`, and `Inf · 0 = NaN` once multiplied by that zero tangent. The outer
`ifelse` sidesteps this: Enzyme's forward-mode `ifelse` selects between the two
branches' *already-computed* tangents based on the primal predicate, so the
poisoned `b^(3//8)` tangent (only ever `Inf` or finite, never itself `NaN`) is
simply discarded when `b ≤ 0`, rather than being combined with a zero weight.
"""
@inline function _vialov_shape(base)
    b = max(base, zero(base))
    return ifelse(b > 0, b^(3//8), zero(b))
end

"""
$(TYPEDSIGNATURES)

Add a radially-symmetric Vialov dome of central thickness `Hc`, radius `L`, centred at `(xc, yc)`:
`H(r) = Hc * max(1 − (r/L)^(4/3), 0)^(3/8)`. Differentiable w.r.t. `xc`, `yc`, `L`, `Hc`.
N.B.: the `(·)^(3/8)` has an infinite slope at the margin (`base → 0⁺`); gradients w.r.t. the centre are steep there but the bulk dominates.
"""
function add_vialov!(H, X, Y, xc, yc, L, Hc)
    @. H += Hc * _vialov_shape(1 - (sqrt((X - xc)^2 + (Y - yc)^2) / L)^(4//3))
    return nothing
end

# Write the physical viscosity field from a log10 field held in `logη`.
set_viscosity_from_log10!(sim, logη) = (@. sim.solidearth.effective_viscosity = 10^logη; nothing)

# Coordinate grids on the same array kind as `ref`. `domain.X`/`domain.Y` are always
# CPU `Matrix`es (they are only promoted for setup), but on a GPU simulation the
# *differentiated* `reconstruct!` broadcasts them against device fields, which cannot
# mix host and device arrays. Dispatch on `ref`'s concrete type — `domain.arraykernel`
# is stored as `::Any`, so using it here would make `reconstruct!` type-unstable and
# trip Enzyme. On CPU (`ref::Array`) this returns the grid untouched (zero copy,
# byte-identical to before); on GPU it copies the const grid onto the device once.
match_array(ref::Array, X) = X
match_array(ref, X) = copyto!(similar(ref, eltype(X), size(X)), X)

# =============================================================================
# Full-field "encoding" (`encoding === nothing`) — direct 2-D viscosity control.
# =============================================================================

# No encoding: θ *is* the full 2-D effective-viscosity field, controlled in log10
# space (the natural, well-conditioned parameterization — physical value 10^θ),
# flattened column-major to match `effective_viscosity`'s `Matrix` layout. This is
# the full-field inversion target of AdjointMode (roadmap Phase 5, Test 3): with a
# reverse sweep the gradient cost is independent of the (grid-sized) parameter
# count, so no dimensionality-reducing encoding is needed. Reuses the same broadcast
# `set_viscosity_from_log10!` that the encoded paths differentiate, so it is
# Enzyme-legal by construction (`reshape` is a view; the `10^` broadcast is the
# proven path).
function reconstruct!(sim, θ, ::Nothing)
    logη = reshape(θ, size(sim.solidearth.effective_viscosity))
    set_viscosity_from_log10!(sim, logη)
    return nothing
end

# The ice-thickness snapshots an encoding writes into (for time interpolation).
ice_snapshots(sim) = sim.bcs.ice_thickness.H_itp.X

# =============================================================================
# Test1Encoding — joint Vialov ice (3 domes, time-varying) + bimodal viscosity.
# =============================================================================

"""
    Test1Encoding(knot_times, radii, visc_amps; scale = ones)

Encoding for Test 1. Fixed config: `knot_times` (K ice-interpolation times),
`radii` (the 3 fixed Vialov radii `Lᵢ`), `visc_amps` (the 2 fixed log10 anomaly
amplitudes, e.g. `(-1, +1)` decades).

θ layout (length `3K + 13`):
`[Hc₁(1:K), Hc₂(1:K), Hc₃(1:K), x₁,y₁, x₂,y₂, x₃,y₃, log10η_bg, μ₁ₓ,μ₁ᵧ,σ₁, μ₂ₓ,μ₂ᵧ,σ₂]`.

`scale` (length `3K + 13`, default all-ones) rescales each parameter before it
is used: the physical value is `θ[i] * scale[i]`. This lets the *optimization*
variable `θ` be dimensionless and O(1) even though the physical parameters span
many orders of magnitude (metres of thickness, metres of position, decades of
viscosity) — essential for L-BFGS conditioning. With the default ones, θ is the
physical parameter vector directly. The mapping is a plain broadcast, so it stays
Enzyme-legal.
"""
struct Test1Encoding{T} <: AbstractEncoding{T}
    knot_times::Vector{T}
    radii::NTuple{3, T}     # TODO: replace 3 by N1
    visc_amps::NTuple{2, T} # TODO: replace 2 by N2
    scale::Vector{T}
end

function Test1Encoding(knot_times::AbstractVector, radii, visc_amps; scale = nothing)
    T = eltype(knot_times)
    np = 3 * length(knot_times) + 13
    s = scale === nothing ? ones(T, np) : convert(Vector{T}, scale)
    length(s) == np || throw(ArgumentError(
        "Test1Encoding scale must have length 3K+13 = $np, got $(length(s))."))
    return Test1Encoding{T}(collect(T, knot_times), NTuple{3, T}(radii),
        NTuple{2, T}(visc_amps), s)
end

nparams(enc::Test1Encoding) = 3 * length(enc.knot_times) + 13

function reconstruct!(sim, θ, enc::Test1Encoding)
    eff = sim.solidearth.effective_viscosity
    X, Y = match_array(eff, sim.domain.X), match_array(eff, sim.domain.Y)
    K = length(enc.knot_times)
    θ = θ .* enc.scale          # dimensionless θ → physical (broadcast, AD-legal)

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
    Test2Encoding(; scale = ones(19))

Encoding for Test 2. θ layout (length 19):
`[log10η_bg, (μₓ,μᵧ,σ,amp)×4, ρ_uppermantle, ρ_litho]`. Ice thickness is not
touched (assumed known). Concrete `Float64` (the inversion-run precision
decision, §1 Precision row).

`scale` (length 19, default all-ones) rescales each parameter before use: the
physical value is `θ[i] * scale[i]`. As with `Test1Encoding`, this lets the
optimization variable `θ` be dimensionless and O(1) even though the physical
parameters span decades of viscosity, metres of anomaly position/width and
thousands of kg/m³ of density — essential for L-BFGS conditioning. The mapping is
a plain broadcast, so it stays Enzyme-legal.
"""
struct Test2Encoding <: AbstractEncoding{Float64}
    scale::Vector{Float64}
end

function Test2Encoding(; scale = nothing)
    s = scale === nothing ? ones(19) : convert(Vector{Float64}, scale)
    length(s) == 19 || throw(ArgumentError(
        "Test2Encoding scale must have length 19, got $(length(s))."))
    return Test2Encoding(s)
end

nparams(::Test2Encoding) = 19

function reconstruct!(sim, θ, enc::Test2Encoding)
    eff = sim.solidearth.effective_viscosity
    X, Y = match_array(eff, sim.domain.X), match_array(eff, sim.domain.Y)
    θ = θ .* enc.scale          # dimensionless θ → physical (broadcast, AD-legal)
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
