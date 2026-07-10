# =============================================================================
# Regularizations and soft bounds.
#
# Each regularization implements `penalty(reg, sim, θ) -> scalar`, added to the
# data misfit in `loss`. `sim` is assumed to already hold the decoded parameters
# (i.e. `reconstruct!` has run), so field-based penalties can read `sim` directly.
# All penalties are plain differentiable array reductions (Enzyme-legal).
#
# `TikhonovReg(target, order; λ, weights)` is the one configurable regularizer,
# with two independent DOFs:
#   - `target`: *what* is penalised — a raw-θ subset (`ThetaTarget`), a decoded
#     field/scalar (`FieldTarget`, same `BoundedQuantity` selectors as
#     `DecodedBounds`), or the implied ice surface at every snapshot
#     (`SurfaceTarget`, `IceLoadInversion`-specific).
#   - `order`: *how* — `Order0()` penalises magnitude `Σ wᵢ xᵢ²` (optional
#     per-component `weights`, so raw-θ magnitude penalties no longer force a
#     single global scale across parameters of very different natural
#     magnitude); `Order1()` penalises the spatial gradient `Σ ‖∇x‖²` via the FD
#     stencils (field/surface targets only — gradients are meaningless on the
#     unstructured `ThetaTarget`).
# `L2Reg`/`SurfaceSmoothnessReg` are thin constructors over the general type,
# kept because they're the two common cases (raw-θ magnitude, surface
# smoothness) and read better at call sites than the fully general form.
# =============================================================================

abstract type AbstractRegularization end

# --- decoded-quantity selectors (shared by FieldTarget and DecodedBounds) ----

"""
Bounded decoded quantities, selected without closures (Enzyme-friendly).
"""
abstract type BoundedQuantity end
struct Log10Viscosity <: BoundedQuantity end
struct UpperMantleDensity <: BoundedQuantity end
struct LithoDensity <: BoundedQuantity end

decoded(::Log10Viscosity, sim) = log10.(sim.solidearth.effective_viscosity)
decoded(::UpperMantleDensity, sim) = sim.solidearth.rho_uppermantle
decoded(::LithoDensity, sim) = sim.solidearth.rho_litho

# --- regularization targets ---------------------------------------------

abstract type AbstractRegTarget end

"""
    ThetaTarget(idx = Colon())

Regularization target: the raw parameter vector `θ` (encoded space), or a
subset of it selected by `idx` (anything that indexes a vector — `Colon()`,
`UnitRange`, `Vector{Int}`). Only meaningful with `Order0()` — `θ` has no
spatial structure to take a gradient of.
"""
struct ThetaTarget{I} <: AbstractRegTarget
    idx::I
end
ThetaTarget() = ThetaTarget(Colon())

"""
    FieldTarget(quantity::BoundedQuantity)

Regularization target: a decoded field or scalar, `decoded(quantity, sim)`.
`quantity` is one of the `BoundedQuantity` selectors shared with
`DecodedBounds` (`Log10Viscosity`, `UpperMantleDensity`, `LithoDensity`).
"""
struct FieldTarget{Q<:BoundedQuantity} <: AbstractRegTarget
    quantity::Q
end

"""
    SurfaceTarget()

Regularization target: the implied ice surface `s = Hₖ + b_ref` at every ice
snapshot `Hₖ` (`IceLoadInversion`), with a fixed reference bed
`b_ref = sim.ref.z_b`. `Order1()` gives the original `SurfaceSmoothnessReg`
penalty.
"""
struct SurfaceTarget <: AbstractRegTarget end

# --- regularization orders ---------------------------------------------

abstract type AbstractRegOrder end

"Magnitude penalty `Σ wᵢ xᵢ²` (optionally weighted)."
struct Order0 <: AbstractRegOrder end

"Gradient penalty `Σ ‖∇x‖²` via the FD stencils (field/surface targets only)."
struct Order1 <: AbstractRegOrder end

_weighted_sumabs2(x, ::Nothing) = sum(abs2, x)
_weighted_sumabs2(x, w) = sum(wi * abs2(xi) for (xi, wi) in zip(x, w))

function _gradient_sumabs2(field, domain)
    sx = similar(field); sy = similar(field)
    dx!(sx, field, domain)
    dy!(sy, field, domain)
    return sum(abs2, sx) + sum(abs2, sy)
end

# --- TikhonovReg -------------------------------------------------------------

"""
    TikhonovReg(target::AbstractRegTarget, order::AbstractRegOrder = Order0();
        λ = 1, weights = nothing)

One configurable Tikhonov-style regularizer; see the module docstring above for
`target`/`order`. `weights`, if given, must match the target's shape/length and
only applies under `Order0()`.
"""
struct TikhonovReg{Ta<:AbstractRegTarget, O<:AbstractRegOrder, T<:Real, W} <: AbstractRegularization
    target::Ta
    order::O
    λ::T
    weights::W
end

TikhonovReg(target::AbstractRegTarget, order::AbstractRegOrder = Order0();
    λ = 1.0, weights = nothing) = TikhonovReg(target, order, λ, weights)

penalty(reg::TikhonovReg, sim, θ) =
    reg.λ * _tikhonov_penalty(reg.target, reg.order, sim, θ, reg.weights)

_tikhonov_penalty(target::ThetaTarget, ::Order0, sim, θ, weights) =
    _weighted_sumabs2(view(θ, target.idx), weights)

_tikhonov_penalty(target::FieldTarget, ::Order0, sim, θ, weights) =
    _weighted_sumabs2(decoded(target.quantity, sim), weights)

function _tikhonov_penalty(target::FieldTarget, ::Order1, sim, θ, weights)
    d = decoded(target.quantity, sim)
    d isa Number && throw(ArgumentError(
        "Order1 (gradient) regularization needs a field target, got the " *
        "scalar quantity $(target.quantity)."))
    return _gradient_sumabs2(d, sim.domain)
end

function _tikhonov_penalty(::SurfaceTarget, ::Order1, sim, θ, weights)
    domain = sim.domain
    snaps = ice_snapshots(sim)
    b = sim.ref.z_b
    s = similar(b)
    acc = zero(eltype(b))
    for Hk in snaps
        @. s = Hk + b
        acc += _gradient_sumabs2(s, domain)
    end
    return acc
end

_tikhonov_penalty(::ThetaTarget, ::Order1, sim, θ, weights) = throw(ArgumentError(
    "Order1 (gradient) regularization is not defined on ThetaTarget — θ has " *
    "no spatial structure to differentiate."))

"""
    L2Reg(λ)

Thin constructor over `TikhonovReg`: magnitude penalty on the full raw
parameter vector, `λ · ‖θ‖²`.
"""
L2Reg(λ) = TikhonovReg(ThetaTarget(), Order0(); λ = λ)

"""
    SurfaceSmoothnessReg(λ)

Thin constructor over `TikhonovReg`: gradient penalty on the implied ice
surface, `λ · Σₖ ‖∇(Hₖ + b_ref)‖²`.
"""
SurfaceSmoothnessReg(λ) = TikhonovReg(SurfaceTarget(), Order1(); λ = λ)

# --- soft decoded bounds -----------------------------------------------------

"""
    DecodedBounds(quantity, lo, hi, λ)

Soft two-sided hinge penalty keeping a decoded quantity within `[lo, hi]`:
`λ · Σ [relu(lo − d)² + relu(d − hi)²]`, where `d = decoded(quantity, sim)` is
either a field or a scalar. `quantity` is a `BoundedQuantity`
(`Log10Viscosity`, `UpperMantleDensity`, `LithoDensity`).
"""
struct DecodedBounds{Q<:BoundedQuantity, T<:Real} <: AbstractRegularization
    quantity::Q
    lo::T
    hi::T
    λ::T
end

_hinge(d, lo, hi) = max(lo - d, zero(d))^2 + max(d - hi, zero(d))^2

function penalty(reg::DecodedBounds, sim, θ)
    d = decoded(reg.quantity, sim)
    if d isa Number
        return reg.λ * _hinge(d, reg.lo, reg.hi)
    else
        return reg.λ * sum(x -> _hinge(x, reg.lo, reg.hi), d)
    end
end
