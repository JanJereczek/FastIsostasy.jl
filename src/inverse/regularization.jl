# =============================================================================
# Regularizations and soft bounds.
#
# Each regularization implements `penalty(reg, sim, θ) -> scalar`, added to the
# data misfit in `loss`. `sim` is assumed to already hold the decoded parameters
# (i.e. `reconstruct!` has run), so field-based penalties can read `sim` directly.
# All penalties are plain differentiable array reductions (Enzyme-legal).
# =============================================================================

abstract type AbstractRegularization end

"""
    L2Reg(λ)

Tikhonov penalty on the parameter vector: `λ · ‖θ‖²`.
"""
struct L2Reg{T<:Real} <: AbstractRegularization
    λ::T
end

penalty(reg::L2Reg, sim, θ) = reg.λ * sum(abs2, θ)

"""
    SurfaceSmoothnessReg(λ)

Penalise a rough implied ice surface for `IceLoadInversion`:
`λ · Σₖ ‖∇(Hₖ + b_ref)‖²`, summed over the ice snapshots `Hₖ`, with a fixed
reference bed `b_ref = sim.ref.z_b`. Uses the existing FD stencils.
"""
struct SurfaceSmoothnessReg{T<:Real} <: AbstractRegularization
    λ::T
end

function penalty(reg::SurfaceSmoothnessReg, sim, θ)
    domain = sim.domain
    snaps = ice_snapshots(sim)
    b = sim.ref.z_b
    sx = similar(b); sy = similar(b); s = similar(b)
    acc = zero(eltype(b))
    for Hk in snaps
        @. s = Hk + b
        dx!(sx, s, domain)
        dy!(sy, s, domain)
        acc += sum(abs2, sx) + sum(abs2, sy)
    end
    return reg.λ * acc
end

# --- soft decoded bounds -----------------------------------------------------

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
