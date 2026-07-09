module FastIsostasyEnzymeExt

# Enzyme-based differentiation of `loss(prob, θ)` (roadmap Phase 2).
#
# This first cut provides the two foundational, model-independent pieces:
#   1. `EnzymeRules.inactive` for `t_computation!` (wall-clock timer in the RHS).
#   2. Forward-mode `EnzymeRules` for planned-FFT application `mul!(Y, plan, X)`.
#
# `gradient!(g, prob, θ, ::TangentMode)` (the go/no-go, needs a shadow `Simulation`)
# and the reverse-mode plan rules (Phase 5) build on top of these; see
# `docs/src/inversion_ad_activity_map.md` for the field-by-field activity reference.

# Everything is reached through `Enzyme` (which re-exports EnzymeCore + EnzymeRules)
# so the extension needs no weakdep beyond Enzyme itself.
using Enzyme: Enzyme, Const, Duplicated, BatchDuplicated,
    DuplicatedNoNeed, BatchDuplicatedNoNeed, Annotation, EnzymeRules
using Enzyme.EnzymeRules: FwdConfig
using LinearAlgebra: mul!
using AbstractFFTs: AbstractFFTs
import FFTW

# Raw FFTW plans (no scale). The custom rule dispatches on these only. Inverse
# transforms on the differentiated path are `FastIsostasy.NormalizedPlan`s (which
# replaced `AbstractFFTs.ScaledPlan` precisely because a `ScaledPlan`'s `Float64`
# `scale` reads as a differentiable leaf and corrupts the tangent); a
# `NormalizedPlan`'s `mul!` calls the inner raw-plan `mul!` (which lands here) then
# scales by a compile-time type-param constant. Any residual `ScaledPlan` is left for
# Enzyme to trace (`lmul!(scale, mul!(y, p.p, x))` — inner raw `mul!` still lands
# here); custom-ruling `ScaledPlan` instead trips Enzyme's `roots_activep` assertion.
const _RawPlan = Union{FFTW.cFFTWPlan, FFTW.rFFTWPlan}

import FastIsostasy: gradient!, loss, AbstractInversion, TangentMode, AdjointMode
import FastIsostasy

# =============================================================================
# 1. Inactive functions (no derivative contribution, no shadow bookkeeping).
# =============================================================================

# `t_computation!` reads the wall clock (`time()`) and `push!`es to the timer
# vectors — pure instrumentation, never read back into active computation
# (roadmap §2 audit). `sim.timer.t = t` is inactive-by-type and needs nothing.
EnzymeRules.inactive(::typeof(FastIsostasy.t_computation!), args...) = nothing

# FFT/convolution plans are fixed linear operators, never differentiated (their
# application is handled by the custom `mul!` rule below). Marking their types
# inactive keeps them `Const` under autodiff and stops `make_zero`/shadow builders
# from allocating meaningless plan shadows — so the `mul!` rule always sees a
# `Const` plan. (roadmap activity map: plans are Const.) `NormalizedPlan` wraps the
# raw inverse plan + an Int-free type-param scale; marking it inactive is belt-and-
# suspenders (the type param already keeps the scale out of the active set).
EnzymeRules.inactive_type(::Type{<:AbstractFFTs.Plan}) = true
EnzymeRules.inactive_type(::Type{<:FastIsostasy.NormalizedPlan}) = true

# =============================================================================
# 2. Forward-mode rule for planned-FFT application `mul!(Y, plan, X)`.
#
# An FFT plan is a fixed *linear* operator P, so `Y = P·X` has Jacobian P itself:
# the tangent is the same transform applied to the input tangent, `dY = P·dX`.
# Restricting `plan::Const{<:AbstractFFTs.Plan}` keeps this from hijacking the
# generic matrix `mul!`. Covers complex `plan_fft`/`plan_ifft` (explicit
# MaxwellMantle path) and `rfft`/`irfft` (ConvolutionPlan, RealMaxwellMantle);
# the transform is linear in every case, so one rule serves all.
# =============================================================================

@inline _shadow(x::Duplicated, ::Int) = x.dval
@inline _shadow(x::BatchDuplicated, b::Int) = x.dval[b]

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(mul!)},
        ::Type{RT},
        Y::Annotation{<:AbstractArray},
        plan::Annotation{<:_RawPlan},
        X::Annotation{<:AbstractArray},
    ) where {RT}

    # The plan is a fixed operator: use its primal value and ignore any shadow.
    # Accepting `plan::Annotation` (not just `Const`) matters because inverse
    # plans are `ScaledPlan`s carrying a `Float64` scale — Enzyme's type analysis
    # can present them as `Duplicated` even though the operator is constant.
    p = plan.val

    # Primal: the actual transform. `mul!` writes `Y.val` in place and preserves
    # `X.val` (out-of-place plan) — the input-preservation AD relies on.
    mul!(Y.val, p, X.val)

    # Tangents: propagate each shadow column through the same linear operator.
    if !(Y isa Const)
        for b in 1:EnzymeRules.width(config)
            dY = _shadow(Y, b)
            if X isa Const
                fill!(dY, zero(eltype(dY)))   # constant input ⇒ zero tangent
            else
                mul!(dY, p, _shadow(X, b))
            end
        end
    end

    # Return in the shape Enzyme requested (`mul!` returns its destination `Y`).
    # Every call site discards the return, so `RT <: Const` is the hot path.
    if RT <: Const || Y isa Const
        return nothing
    elseif RT <: DuplicatedNoNeed || RT <: BatchDuplicatedNoNeed
        return EnzymeRules.width(config) == 1 ? _shadow(Y, 1) :
            ntuple(b -> _shadow(Y, b), EnzymeRules.width(config))
    elseif RT <: Duplicated
        return Duplicated(Y.val, _shadow(Y, 1))
    elseif RT <: BatchDuplicated
        return BatchDuplicated(Y.val,
            ntuple(b -> _shadow(Y, b), EnzymeRules.width(config)))
    else
        return nothing
    end
end

# =============================================================================
# 3. `gradient!` — forward-mode (TangentMode).
#
# `∇_θ loss(prob, θ)` by seeding one θ direction at a time and reading the tangent
# of the scalar loss. The whole `prob` (which `loss` mutates in place) is shadowed
# once via `make_zero` and re-zeroed per direction. `set_runtime_activity` is
# required: the explicit `update_dudt!` reuses a complex buffer as both fft-input and
# ifft-output, which static activity analysis mishandles (see the activity-map doc).
# Cost is one forward pass per θ component — affordable only for low-dim (encoded) θ,
# which `TangentMode` enforces.
# =============================================================================

function gradient!(g, prob::AbstractInversion, θ)
    return gradient!(g, prob, θ, prob.diffmode)
end

function gradient!(g, prob::AbstractInversion, θ, ::TangentMode)
    length(g) == length(θ) || throw(DimensionMismatch(
        "gradient buffer length $(length(g)) ≠ θ length $(length(θ))"))
    mode = Enzyme.set_runtime_activity(Enzyme.Forward)
    dprob = Enzyme.make_zero(prob)
    dθ = zero(θ)
    for i in eachindex(θ)
        Enzyme.remake_zero!(dprob)               # reset the shadow (skips the
                                                 # immutable-nonzero check that
                                                 # `make_zero!` trips on plans)
        fill!(dθ, zero(eltype(dθ)))
        dθ[i] = one(eltype(dθ))
        g[i] = only(Enzyme.autodiff(mode, loss,
            Duplicated(prob, dprob), Duplicated(θ, dθ)))
    end
    return g
end

function gradient!(g, prob::AbstractInversion, θ, ::AdjointMode)
    error("Reverse-mode `gradient!` (AdjointMode) requires " *
          "FastIsostasyCheckpointingExt (roadmap Phase 5).")
end

end # module
