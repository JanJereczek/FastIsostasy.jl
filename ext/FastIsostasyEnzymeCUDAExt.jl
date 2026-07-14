module FastIsostasyEnzymeCUDAExt

# Enzyme rules for CUFFT plan application on the GPU (roadmap Phase 6).
#
# `FastIsostasyEnzymeExt` supplies the custom `mul!(Y, plan, X)` rules that let
# Enzyme differentiate the planned FFTs on the differentiated path, but those rules
# are dispatched on the *FFTW* plan types (`cFFTWPlan`/`rFFTWPlan`). On a GPU
# simulation the plans are `CUDA.CUFFT.CuFFTPlan`s instead — still
# `AbstractFFTs.Plan`s (so `inactive_type` keeps them `Const`), but without a `mul!`
# rule Enzyme would try to differentiate the raw CUFFT call. This extension adds the
# same rules for the CUFFT plan types; it loads only when *both* `Enzyme` and `CUDA`
# are present.
#
# The maths is identical to the CPU rules (an FFT plan is a fixed linear operator
# `P`: tangent `dY = P·dX`, adjoint `X̄ += Pᴴ·Ȳ`), and the adjoint helpers are
# array-generic — `conj`/`real`/`similar`/`mul!`/`AbstractFFTs.bfft`/`rfft` all
# dispatch to CUFFT on `CuArray`s — so we only re-declare the dispatch on the CUFFT
# plan types. Plan-kind discriminators use the `CuFFTPlan{Tout,Tin,…}` type params
# (param 1 = output eltype, param 2 = input eltype):
#   • complex fft/ifft : `{<:Complex, <:Complex}`
#   • rfft  (real→cplx): `{<:Complex, <:Real}`
#   • brfft (cplx→real): `{<:Real, <:Complex}`

using Enzyme: Enzyme, Const, Duplicated, BatchDuplicated,
    DuplicatedNoNeed, BatchDuplicatedNoNeed, Annotation, EnzymeRules
using Enzyme.EnzymeRules: FwdConfig
using LinearAlgebra: mul!, dot
using AbstractFFTs: AbstractFFTs
using CUDA: CUDA, CuArray
import FastIsostasy

const _CuPlan = CUDA.CUFFT.CuFFTPlan
const _CuCplx = CUDA.CUFFT.CuFFTPlan{<:Complex, <:Complex}   # complex fft / ifft
const _CuRfft = CUDA.CUFFT.CuFFTPlan{<:Complex, <:Real}      # rfft  (real → complex)
const _CuBrfft = CUDA.CUFFT.CuFFTPlan{<:Real, <:Complex}     # brfft (complex → real)

# Belt-and-suspenders: CuFFTPlans are `AbstractFFTs.Plan`s, so the Enzyme ext's
# `inactive_type(::Type{<:AbstractFFTs.Plan})` already covers them; re-declaring is
# harmless and keeps this extension self-contained.
EnzymeRules.inactive_type(::Type{<:_CuPlan}) = true

# =============================================================================
# Forward-mode: `dY = P·dX` (one rule covers every CUFFT plan kind — forward mode
# of a linear operator is just the operator applied to the tangent).
# =============================================================================

@inline _shadow(x::Duplicated, ::Int) = x.dval
@inline _shadow(x::BatchDuplicated, b::Int) = x.dval[b]

# =============================================================================
# Forward-mode rule for `sumabs2` on a `CuArray`.
#
# Enzyme cannot differentiate a GPU reduction: unshielded, its `cufunction` rule
# has no method for the `partial_mapreduce_grid{abs2, add_sum}` kernel. Shielding
# it behind this rule fixes that — but the rule *body* must also avoid CUDA's
# `mapreduce`, because `mapreducedim!` → `launch_configuration` →
# `cuOccupancyMaxPotentialBlockSize` calls back into Julia through a `@cfunction`
# (`shmem_cint`) that segfaults when invoked from inside Enzyme's JIT frame.
#
# So both value and tangent go through CUBLAS `dot`, which touches neither path:
#   f(x) = Σ|xᵢ|²  = dot(x, x)
#   df   = 2·Re⟨x, dx⟩ = 2·real(dot(x, dx))
# (`dot` conjugates its first argument, so this is correct for complex `x` too.)
#
# `vec` is essential, not cosmetic: only `dot(::StridedCuVector{<:CublasFloat}, …)`
# reaches the real `cublasDdot`. `dot` on a `CuMatrix` falls back to cuBLAS's
# generic `AnyCuArray` reduction (linalg.jl), which hand-rolls a kernel and calls
# `launch_configuration(...; shmem = …)` — i.e. the very occupancy callback we are
# avoiding, and it segfaults here exactly like `sum` does. Our inputs are freshly
# allocated contiguous arrays, so `vec` is a free reshape.
#
# Dispatched on `CuArray` only — on CPU `sumabs2` keeps its native, already
# validated `sum(abs2, ·)` derivative.
# =============================================================================

@inline _blas_sumabs2(x) = (xv = vec(x); real(dot(xv, xv)))
@inline _blas_reinner(x, y) = real(dot(vec(x), vec(y)))

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(FastIsostasy.sumabs2)},
        ::Type{RT},
        x::Annotation{<:CuArray},
    ) where {RT}

    p = _blas_sumabs2(x.val)
    RT <: Const && return nothing

    if EnzymeRules.width(config) == 1
        dp = x isa Const ? zero(p) : 2 * _blas_reinner(x.val, _shadow(x, 1))
        RT <: DuplicatedNoNeed && return dp
        RT <: Duplicated && return Duplicated(p, dp)
    else
        dps = ntuple(EnzymeRules.width(config)) do b
            x isa Const ? zero(p) : 2 * _blas_reinner(x.val, _shadow(x, b))
        end
        RT <: BatchDuplicatedNoNeed && return dps
        RT <: BatchDuplicated && return BatchDuplicated(p, dps)
    end
    return nothing
end

# --- `totalsum` (Σxᵢ) ---------------------------------------------------------
#
# `sum(x) = dot(ones, x)`, so both value and tangent stay on the `cublasDdot` path.
# The `ones` vector is cached per (eltype, length) and **built on the host**
# (`CuArray(ones(...))` = a plain H2D memcpy): `CUDA.ones`/`fill!` would be another
# kernel launch, and kernel launches inside a rule body are exactly what we are
# trying to avoid. The cache is opaque to Enzyme (rule bodies are not
# differentiated), so mutating it here is safe.
const _ONES = Dict{Tuple{DataType, Int}, Any}()

function _ones_like(x::CuArray)
    T, n = eltype(x), length(x)
    return get!(() -> CuArray(ones(T, n)), _ONES, (T, n))::CuArray{T, 1}
end

@inline _blas_total(x) = dot(_ones_like(x), vec(x))

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(FastIsostasy.totalsum)},
        ::Type{RT},
        x::Annotation{<:CuArray},
    ) where {RT}

    p = _blas_total(x.val)
    RT <: Const && return nothing

    if EnzymeRules.width(config) == 1
        dp = x isa Const ? zero(p) : _blas_total(_shadow(x, 1))
        RT <: DuplicatedNoNeed && return dp
        RT <: Duplicated && return Duplicated(p, dp)
    else
        dps = ntuple(EnzymeRules.width(config)) do b
            x isa Const ? zero(p) : _blas_total(_shadow(x, b))
        end
        RT <: BatchDuplicatedNoNeed && return dps
        RT <: BatchDuplicated && return BatchDuplicated(p, dps)
    end
    return nothing
end

# --- `inner` (⟨a, b⟩) ---------------------------------------------------------
#
# Bilinear: d⟨a,b⟩ = ⟨da,b⟩ + ⟨a,db⟩. In `apply_bc!` the weights `a = bc.W` are
# `Const`, so in practice only the second term survives — but both are handled.
# Enzyme has no derivative for `dot(::CuArray, ::CuArray)` at all
# (`EnzymeNoDerivativeError`), which is why this shim exists.
@inline _blas_inner(a, b) = dot(vec(a), vec(b))

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(FastIsostasy.inner)},
        ::Type{RT},
        a::Annotation{<:CuArray},
        b::Annotation{<:CuArray},
    ) where {RT}

    p = _blas_inner(a.val, b.val)
    RT <: Const && return nothing

    dpart(bi) = begin
        d = zero(p)
        a isa Const || (d += _blas_inner(_shadow(a, bi), b.val))
        b isa Const || (d += _blas_inner(a.val, _shadow(b, bi)))
        d
    end

    if EnzymeRules.width(config) == 1
        dp = dpart(1)
        RT <: DuplicatedNoNeed && return dp
        RT <: Duplicated && return Duplicated(p, dp)
    else
        dps = ntuple(dpart, EnzymeRules.width(config))
        RT <: BatchDuplicatedNoNeed && return dps
        RT <: BatchDuplicated && return BatchDuplicated(p, dps)
    end
    return nothing
end

function EnzymeRules.forward(
        config::FwdConfig,
        ::Const{typeof(mul!)},
        ::Type{RT},
        Y::Annotation{<:AbstractArray},
        plan::Annotation{<:_CuPlan},
        X::Annotation{<:AbstractArray},
    ) where {RT}

    p = plan.val
    mul!(Y.val, p, X.val)

    if !(Y isa Const)
        for b in 1:EnzymeRules.width(config)
            dY = _shadow(Y, b)
            if X isa Const
                fill!(dY, zero(eltype(dY)))
            else
                mul!(dY, p, _shadow(X, b))
            end
        end
    end

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
# Reverse-mode: `X̄ += Pᴴ·Ȳ`, output cotangent consumed. Shared augmented-primal
# (just run the transform), three `reverse` methods by plan kind. The adjoint
# accumulation helpers are the same maths as the CPU ext (replicated here since
# extension modules can't share internals), array-generic → run on `CuArray`.
# =============================================================================

@inline _rev_shadows(x::Duplicated) = (x.dval,)
@inline _rev_shadows(x::BatchDuplicated) = x.dval

@inline function _zero_output!(Yshs)
    for Ȳ in Yshs
        fill!(Ȳ, zero(eltype(Ȳ)))
    end
end

# complex: `Pᴴ·Ȳ = conj(P·conj(Ȳ))`
function _accum_cplan_adjoint!(Xbar, plan, Ȳ)
    cy = conj.(Ȳ)
    tmp = similar(cy)
    mul!(tmp, plan, cy)
    @. Xbar += conj(tmp)
    return nothing
end

# rfft adjoint: `X̄ += Re(bfft(zeropadₙ(Ȳ)))`
function _accum_rfft_adjoint!(Xbar, Ȳ)
    N, ncol = size(Xbar)
    padded = CUDA.zeros(eltype(Ȳ), N, ncol)
    @views padded[1:size(Ȳ, 1), :] .= Ȳ
    z = AbstractFFTs.bfft(padded)
    @. Xbar += real(z)
    return nothing
end

# brfft adjoint: `X̄ += D ⊙ rfft(Ȳ)`, `D` doubling the interior dim-1 rows.
function _accum_brfft_adjoint!(Xbar, Ȳ)
    N = size(Ȳ, 1)
    R = AbstractFFTs.rfft(Ȳ)
    m = size(R, 1)
    @views Xbar[1, :] .+= R[1, :]
    hi = iseven(N) ? m - 1 : m
    hi >= 2 && (@views Xbar[2:hi, :] .+= 2 .* R[2:hi, :])
    iseven(N) && (@views Xbar[m, :] .+= R[m, :])
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(mul!)},
        ::Type{RT},
        Y::Annotation{<:AbstractArray},
        plan::Const{<:_CuPlan},
        X::Annotation{<:AbstractArray},
    ) where {RT}
    mul!(Y.val, plan.val, X.val)
    primal = EnzymeRules.needs_primal(config) ? Y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Y.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(mul!)}, ::Type{RT}, tape,
        Y::Annotation{<:AbstractArray}, plan::Const{<:_CuCplx},
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
        Y::Annotation{<:AbstractArray}, plan::Const{<:_CuRfft},
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
        Y::Annotation{<:AbstractArray}, plan::Const{<:_CuBrfft},
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

end # module
