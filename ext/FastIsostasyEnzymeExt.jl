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

import FastIsostasy: gradient!, loss_and_gradient!, loss, AbstractInversion,
    TangentMode, AdjointMode
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
# 2b. Reverse-mode rules for planned-FFT application `mul!(Y, plan, X)` (Phase 5).
#
# For a fixed linear operator `P`, `Y = P·X` has pullback `X̄ += Pᴴ·Ȳ`, and the
# output cotangent is consumed (`Ȳ` zeroed — `mul!` fully overwrites `Y`, always
# with distinct dest/src buffers). The augmented-primal is shared (just run the
# transform, no tape — the Const plan is available again in `reverse`); only the
# adjoint `Pᴴ` differs by plan kind, so there are three `reverse` methods:
#
#  • complex `cFFTWPlan` (`fft`/`bfft`, the `update_dudt!` path): each raw DFT
#    operator is *complex-symmetric* (`Wᵀ = W`), so `Pᴴ = conj(P)` and
#    `Pᴴ·Ȳ = conj(P·conj(Ȳ))` — the *same* plan, no separate transform. Serves both
#    the forward `W` and the raw inverse `W̄` inside a `NormalizedPlan` (whose
#    `y .*= scale` is differentiated natively).
#  • real forward `rfft` (`rFFTWPlan{Float64}`): `R = S₁·F` projects dim-1 to
#    `m = N÷2+1` rows, so `Rᴴ(Ȳ) = Re(bfft(zeropadₙ(Ȳ)))` — zero-pad the dropped
#    rows, 2-D complex `bfft`, take the real part (`X` is real).
#  • real backward `brfft` (`rFFTWPlan{<:Complex}`, the raw plan inside the irfft
#    `NormalizedPlan`): `Bᴴ(Z̄) = D ⊙ rfft(Z̄)`, where `D` *doubles* the interior
#    dim-1 frequencies (DC and, for even `N`, Nyquist stay 1) — the transpose of the
#    hermitian doubling `brfft` applies on the way forward.
#
# The rfft/brfft adjoints need the *complementary* transform (`bfft`/`rfft`), built
# on the fly via `AbstractFFTs` here (correctness first; a threaded plan is a later
# optimisation). CUFFT plans are `AbstractFFTs.Plan`s too, so the same rules apply
# on GPU (Phase 6 verification).
# =============================================================================

const _RfftPlan = FFTW.rFFTWPlan{Float64}       # real → complex half (forward rfft)
const _BrfftPlan = FFTW.rFFTWPlan{<:Complex}    # complex half → real (backward brfft)

@inline _rev_shadows(x::Duplicated) = (x.dval,)
@inline _rev_shadows(x::BatchDuplicated) = x.dval

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(mul!)},
        ::Type{RT},
        Y::Annotation{<:AbstractArray},
        plan::Const{<:_RawPlan},
        X::Annotation{<:AbstractArray},
    ) where {RT}

    # Forward sweep: run the actual transform (writes `Y.val`, preserves `X.val`).
    mul!(Y.val, plan.val, X.val)

    primal = EnzymeRules.needs_primal(config) ? Y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Y.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

# --- adjoint kernels ---------------------------------------------------------

# complex `Pᴴ·Ȳ = conj(P·conj(Ȳ))` accumulated into `Xbar`.
function _accum_cplan_adjoint!(Xbar, plan, Ȳ)
    cy = conj.(Ȳ)
    tmp = similar(cy)
    mul!(tmp, plan, cy)
    @. Xbar += conj(tmp)
    return nothing
end

# rfft adjoint: `X̄ += Re(bfft(zeropadₙ(Ȳ)))`. `Ȳ` is `m×n` complex, `Xbar` is
# `N×n` real (`N = size(Xbar,1)`, `m = N÷2+1`).
function _accum_rfft_adjoint!(Xbar, Ȳ)
    N, ncol = size(Xbar)
    padded = zeros(eltype(Ȳ), N, ncol)
    @views padded[1:size(Ȳ, 1), :] .= Ȳ
    z = AbstractFFTs.bfft(padded)
    @. Xbar += real(z)
    return nothing
end

# brfft adjoint: `X̄ += D ⊙ rfft(Ȳ)`, `D` doubling the interior dim-1 rows. `Ȳ` is
# `N×n` real, `Xbar` is `m×n` complex (`N = size(Ȳ,1)`).
function _accum_brfft_adjoint!(Xbar, Ȳ)
    N = size(Ȳ, 1)
    R = AbstractFFTs.rfft(Ȳ)                 # m×n complex, m = N÷2+1
    m = size(R, 1)
    @views Xbar[1, :] .+= R[1, :]            # DC row: not doubled
    hi = iseven(N) ? m - 1 : m               # last interior row
    hi >= 2 && (@views Xbar[2:hi, :] .+= 2 .* R[2:hi, :])
    iseven(N) && (@views Xbar[m, :] .+= R[m, :])   # Nyquist row: not doubled
    return nothing
end

# --- reverse methods (one per plan kind) -------------------------------------

# Zero the (fully-overwritten) output cotangents after the adjoint accumulation,
# for every batch member.
@inline function _zero_output!(Yshs)
    for Ȳ in Yshs
        fill!(Ȳ, zero(eltype(Ȳ)))
    end
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(mul!)}, ::Type{RT}, tape,
        Y::Annotation{<:AbstractArray}, plan::Const{<:FFTW.cFFTWPlan},
        X::Annotation{<:AbstractArray}) where {RT}
    if !(Y isa Const)
        Yshs = _rev_shadows(Y)
        if !(X isa Const)
            Xshs = _rev_shadows(X)
            for b in 1:EnzymeRules.width(config)
                _accum_cplan_adjoint!(Xshs[b], plan.val, Yshs[b])
            end
        end
        _zero_output!(Yshs)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(mul!)}, ::Type{RT}, tape,
        Y::Annotation{<:AbstractArray}, plan::Const{<:_RfftPlan},
        X::Annotation{<:AbstractArray}) where {RT}
    if !(Y isa Const)
        Yshs = _rev_shadows(Y)
        if !(X isa Const)
            Xshs = _rev_shadows(X)
            for b in 1:EnzymeRules.width(config)
                _accum_rfft_adjoint!(Xshs[b], Yshs[b])
            end
        end
        _zero_output!(Yshs)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(mul!)}, ::Type{RT}, tape,
        Y::Annotation{<:AbstractArray}, plan::Const{<:_BrfftPlan},
        X::Annotation{<:AbstractArray}) where {RT}
    if !(Y isa Const)
        Yshs = _rev_shadows(Y)
        if !(X isa Const)
            Xshs = _rev_shadows(X)
            for b in 1:EnzymeRules.width(config)
                _accum_brfft_adjoint!(Xshs[b], Yshs[b])
            end
        end
        _zero_output!(Yshs)
    end
    return (nothing, nothing, nothing)
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
#
# `forward_predict!` only has an Enzyme-legal path for `EulerIntegrator` (the direct-Euler
# loop, roadmap §4 note (a)); adaptive algorithms fall through to the stateful
# `TableauIntegratorState`, whose `Vector{Matrix}` stage buffers overflow Enzyme's static
# type analysis (`EnzymeNoTypeError`, opaque unless you already know this). Guard
# up front with the documented restriction instead of surfacing that error.
# =============================================================================

_require_fieuler(prob) = prob.sim.opts.diffeq.alg isa FastIsostasy.EulerIntegrator || error(
    "TangentMode v1 is fixed-step only: gradient!/loss_and_gradient! require " *
    "prob.sim.opts.diffeq.alg isa EulerIntegrator (got " *
    "$(typeof(prob.sim.opts.diffeq.alg))). Adaptive algorithms build the " *
    "TableauIntegratorState inside forward_predict!, which Enzyme's static type analysis " *
    "cannot handle (surfaces as a cryptic EnzymeNoTypeError instead).")

function gradient!(g, prob::AbstractInversion, θ, ::TangentMode)
    _require_fieuler(prob)
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

# AdjointMode (reverse) lives in `FastIsostasyCheckpointingExt` (triggered by
# `Checkpointing` + `Enzyme`); with only Enzyme loaded, AdjointMode falls to the
# core mode-aware fallback in problem.jl.

# =============================================================================
# 4. `loss_and_gradient!` — forward-mode (TangentMode), primal along for free.
#
# Same seeding loop as `gradient!`, but each pass uses `ForwardWithPrimal` instead
# of `Forward`: `loss(prob, θ)` doesn't depend on which θ-direction is seeded, so
# the primal read off any one pass equals `loss(prob, θ)` itself — no extra forward
# run is needed to also get the objective value (what a separate `loss`/`gradient!`
# pair would cost, one extra pass per optimizer iteration).
# =============================================================================

function loss_and_gradient!(g, prob::AbstractInversion, θ, ::TangentMode)
    _require_fieuler(prob)
    length(g) == length(θ) || throw(DimensionMismatch(
        "gradient buffer length $(length(g)) ≠ θ length $(length(θ))"))
    mode = Enzyme.set_runtime_activity(Enzyme.ForwardWithPrimal)
    dprob = Enzyme.make_zero(prob)
    dθ = zero(θ)
    l = zero(eltype(θ))
    for i in eachindex(θ)
        Enzyme.remake_zero!(dprob)
        fill!(dθ, zero(eltype(dθ)))
        dθ[i] = one(eltype(dθ))
        dval, val = Enzyme.autodiff(mode, loss,
            Duplicated(prob, dprob), Duplicated(θ, dθ))
        g[i] = dval
        l = val
    end
    return l
end

end # module
