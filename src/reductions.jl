# =============================================================================
# Named reduction shims.
#
# These are deliberately trivial wrappers around `sum`/`dot`. They exist because a
# reduction is the one operation Enzyme cannot differentiate on a `CuArray`, and a
# *named* function is what a custom `EnzymeRules` method can dispatch on.
# `FastIsostasyEnzymeCUDAExt` attaches a forward rule to each, dispatched on
# `<:CuArray`; on CPU no rule matches and Enzyme differentiates the plain
# `sum`/`dot` natively, exactly as before (these wrappers are byte-identical to the
# code they replaced).
#
# Why a rule is needed at all, and why the rule bodies look the way they do
# (roadmap §7):
#   • Unshielded, a GPU reduction fails outright — Enzyme's `cufunction` rule has
#     no method for the `partial_mapreduce_grid` kernel, and `dot(::CuArray, ...)`
#     raises `EnzymeNoDerivativeError`.
#   • A rule whose *body* calls `sum` (or `dot` on a `CuMatrix`) **segfaults**:
#     both route through `launch_configuration(...; shmem = ...)`, which calls back
#     into Julia via a `@cfunction` (`shmem_cint`) that does not survive Enzyme's
#     JIT frame.
#   • Only `dot(::StridedCuVector{<:CublasFloat}, ::StridedCuVector)` reaches the
#     real `cublasDdot`, which takes neither path. Hence every rule body reduces
#     via `dot` on `vec`'d arguments.
#
# Keep these as thin as possible: any logic added here has to be mirrored in the
# extension's rules.
# =============================================================================

"""
    sumabs2(x)

`sum(abs2, x)`. Named so it can carry a GPU AD rule (see the header of
`src/reductions.jl`); on CPU it is exactly `sum(abs2, x)`.
"""
sumabs2(x) = sum(abs2, x)

"""
    totalsum(x)

`sum(x)`. Named so it can carry a GPU AD rule (see the header of
`src/reductions.jl`); on CPU it is exactly `sum(x)`.
"""
totalsum(x) = sum(x)

"""
    inner(a, b)

`dot(a, b)`. Named so it can carry a GPU AD rule (see the header of
`src/reductions.jl`); on CPU it is exactly `dot(a, b)`.
"""
inner(a, b) = dot(a, b)
