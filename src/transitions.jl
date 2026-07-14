# =============================================================================
# Grounding-line / load transitions.
#
# The forward model contains several non-smooth switches at the grounding line
# and the ocean margin: `max(x, 0)` (height above floatation, water column),
# `min(x, 0)` (flotation correction) and Heaviside thresholds `x > 0` (grounded
# and ocean masks). These are exact but non-differentiable, which is a problem
# for gradient-based inversion.
#
# `AbstractTransition` selects how those switches are evaluated:
#   - `SharpTransition` (default): the exact `max`/`min`/`>` operators. Zero
#     cost, byte-identical to the original model, Boolean masks.
#   - `SmoothTransition(ε)`: branch-free C∞ approximations with a length scale
#     `ε` (metres of height-above-flotation). Masks become floating-point.
#
# Scalar primitives are broadcast by the callers; dispatch happens at the
# array-operation level (on the transition type), never per element.
# =============================================================================

"""
    AbstractTransition

Supertype of the transition traits, which decide how the model's non-smooth
switches — grounded/ocean masks, the `max(0, ·)` clamps in the water column — are
evaluated. Stored on [`SolverOptions`](@ref).

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

# --- scalar primitives -------------------------------------------------------

# max(x, 0)
srelu(x, eps) = (x + sqrt(x * x + eps * eps)) / 2
# min(x, 0)
snegrelu(x, eps) = (x - sqrt(x * x + eps * eps)) / 2
# Heaviside (x > 0), smoothed to [0, 1]
sheaviside(x, eps) = (one(x) + x / sqrt(x * x + eps * eps)) / 2
