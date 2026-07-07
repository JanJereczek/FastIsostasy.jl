# =============================================================================
# Differentiation mode.
#
# These types carry *policy* that the AD engine (loaded as an extension) needs
# but that the core package should not depend on: which Enzyme mode to use, the
# tangent batch size / checkpoint schedule, and whether a low-dimensional
# encoding is required. They deliberately do NOT reference Enzyme, so the core
# stays AD-free; the extension maps them onto `Enzyme.Forward` / `Enzyme.Reverse`.
# =============================================================================

abstract type AbstractDiffMode end

"""
    TangentMode(; batch = 8)

Forward-mode (tangent) differentiation. Cost scales with the number of
parameters, so it **requires a low-dimensional encoding**. `batch` is the number
of tangent directions propagated together (chunking).
"""
struct TangentMode <: AbstractDiffMode
    batch::Int
end
TangentMode(; batch::Int = 8) = TangentMode(batch)

"""
    AdjointMode(; checkpoint_every = 10)

Reverse-mode (adjoint) differentiation with periodic checkpointing. Cost is
roughly independent of the parameter dimension, so full 2-D fields can be
inverted without an encoding. `checkpoint_every` is the number of save/output
intervals between stored state snapshots.
"""
struct AdjointMode <: AbstractDiffMode
    checkpoint_every::Int
end
AdjointMode(; checkpoint_every::Int = 10) = AdjointMode(checkpoint_every)

# Does this mode require the problem to carry a (low-dimensional) encoding?
requires_encoding(::TangentMode) = true
requires_encoding(::AdjointMode) = false
