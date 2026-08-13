"""
    AbstractTransition

Supertype of the transition traits, which decide how the model's non-smooth
switches — grounded/ocean masks, the `max(0, ·)` clamps in the water column — are
evaluated. Stored on [`SolverOptions`](@ref).

The scalar primitives are broadcast by the callers, so dispatch happens at the
array-operation level, never per element.

[`SharpTransition`](@ref) (the default) keeps the exact `max`/Heaviside behaviour
at zero cost. [`SmoothTransition`](@ref) replaces them by `ε`-smoothed
counterparts, which is what makes the model **differentiable** across the
grounding line; the masks then become eltype-`T` arrays instead of `Bool`.
"""
abstract type AbstractTransition end

"""
    SharpTransition()

Exact, non-differentiable grounding-line switches (`max`, `min`, `>`). Default;
produces Boolean masks and leaves the forward model unchanged.
"""
struct SharpTransition <: AbstractTransition end

"""
    SmoothTransition(ε)

Differentiable grounding-line switches with transition length scale `ε` (in
metres of height-above-flotation). Uses the branch-free approximations

    max(x, 0) ≈ (x + √(x² + ε²)) / 2
    min(x, 0) ≈ (x − √(x² + ε²)) / 2
    (x > 0)   ≈ (1 + x / √(x² + ε²)) / 2

which are symmetric, GPU-friendly and converge to the exact operators as ε → 0
(error confined to |x| ≲ ε). Masks become floating-point fields in [0, 1].
"""
struct SmoothTransition{T<:Real} <: AbstractTransition
    eps::T
end

# --- scalar primitives (broadcast into GPU kernels, hence `@inline`) ----------

# max(x, 0)
@inline srelu(x, eps) = (x + sqrt(x * x + eps * eps)) / 2
# min(x, 0)
@inline snegrelu(x, eps) = (x - sqrt(x * x + eps * eps)) / 2
# Heaviside (x > 0), smoothed to [0, 1]
@inline sheaviside(x, eps) = (one(x) + x / sqrt(x * x + eps * eps)) / 2
