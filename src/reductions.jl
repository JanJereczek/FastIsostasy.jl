# =============================================================================
# Named reduction shims.
#
# Thin wrappers around `sum`/`dot` so GPU AD rules can dispatch on a named
# function. Enzyme cannot differentiate reductions on `CuArray`, so the
# FastIsostasyEnzymeCUDAExt provides forward rules for these names. On CPU the
# wrappers are byte-identical to the underlying `sum`/`dot`.
# =============================================================================

"""
$(TYPEDSIGNATURES)

`sum(abs2, x)`. Named so it can carry a GPU AD rule (see the header of
`src/reductions.jl`); on CPU it is exactly `sum(abs2, x)`.
"""
sumabs2(x) = sum(abs2, x)

"""
$(TYPEDSIGNATURES)

`sum(x)`. Named so it can carry a GPU AD rule (see the header of
`src/reductions.jl`); on CPU it is exactly `sum(x)`.
"""
totalsum(x) = sum(x)

"""
$(TYPEDSIGNATURES)

`dot(a, b)`. Named so it can carry a GPU AD rule (see the header of
`src/reductions.jl`); on CPU it is exactly `dot(a, b)`.
"""
inner(a, b) = dot(a, b)
