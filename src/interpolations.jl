# Validate a time axis at construction: `ArgumentError` rather than `@assert`, since
# these check *user input* and `@assert` may be elided. `issorted` is not decoration —
# `interpolate`/`interpolate!` read the bounds as `t[1]`/`t[end]` and locate the
# bracketing interval with `searchsortedfirst`, both of which need a sorted axis.
function _check_time_axis(t, n_values::Int, valname::AbstractString, ctor::Symbol)
    length(t) == n_values || throw(DimensionMismatch(
        "$ctor: `t` has $(length(t)) entries but `$valname` has $n_values."))
    isempty(t) && throw(ArgumentError("$ctor: `t` must not be empty."))
    issorted(t) || throw(ArgumentError(
        "$ctor: `t` must be sorted in increasing order."))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Define the time interpolation of a scalar variable.

# Fields
$(TYPEDFIELDS)
"""
struct TimeInterpolation0D{T<:AbstractFloat}
    "a sorted vector of time points at which the variable is defined"
    t::Vector{T}
    "a vector of variable values corresponding to `t`"
    y::Vector{T}
    """
    whether to clamp to the end values outside `[t[1], t[end]]` instead of throwing
    """
    flat_bc::Bool
end

function TimeInterpolation0D(t, y; flat_bc = false)
    _check_time_axis(t, length(y), "y", :TimeInterpolation0D)
    return TimeInterpolation0D(t, y, flat_bc)
end

"""
$(TYPEDSIGNATURES)

Define the time interpolation of an array variable.
"""
mutable struct TimeInterpolation2D{T,M}
    t::Vector{T}
    X::Vector{M}
    flat_bc::Bool
end

function TimeInterpolation2D(t, X; flat_bc = false, backend = nothing)
    _check_time_axis(t, length(X), "X", :TimeInterpolation2D)
    if backend !== nothing
        return TimeInterpolation2D(t, kernelpromote(X, backend), flat_bc)
    else
        return TimeInterpolation2D(t, X, flat_bc)
    end
end

"""
$(TYPEDSIGNATURES)

Interpolate a timeseries at a given time `t_out` using the interpolation object `itp`.
"""
function interpolate(t_out, itp::TimeInterpolation0D)
    t = itp.t
    if t_out < t[1]
        itp.flat_bc || throw(ErrorException("TimeInterpolation0D out of range."))
        return itp.y[1]
    elseif t_out > t[end]
        itp.flat_bc || throw(ErrorException("TimeInterpolation0D out of range."))
        return itp.y[end]
    end
    # One binary search locates the node (exact hit) or the interval above it, so
    # this runs in O(log n) rather than the three O(n) scans a `minimum`/`maximum`/
    # `∈` chain would cost on every call. The axis is sorted by construction, see
    # `_check_time_axis`.
    i = searchsortedfirst(t, t_out)
    t[i] == t_out && return itp.y[i]
    i -= 1
    return itp.y[i] +
           (itp.y[i+1] - itp.y[i]) / (t[i+1] - t[i]) * (t_out - t[i])
end

"""
$(TYPEDSIGNATURES)

Interpolate a time-dependent field (in-place) at a given time `t_out` using the interpolation object `itp`.
"""
function interpolate!(X_out::M, t::T, ti::TimeInterpolation2D{T,M}) where {T,M}
    tv = ti.t
    if t < tv[1]
        ti.flat_bc || throw(ErrorException("TimeInterpolation2D out of range."))
        X_out .= ti.X[1]
        return nothing
    elseif t > tv[end]
        ti.flat_bc || throw(ErrorException("TimeInterpolation2D out of range."))
        X_out .= ti.X[end]
        return nothing
    end
    # This runs on every RHS evaluation (via `apply_bc!(H_ice, t, …)`), so it must
    # not scan the time axis: one binary search, then either an exact node or the
    # bracketing interval. See `interpolate(::TimeInterpolation0D)`.
    i = searchsortedfirst(tv, t)
    if tv[i] == t
        X_out .= ti.X[i]
    else
        i -= 1
        @. X_out = ti.X[i] + (ti.X[i+1] - ti.X[i]) / (tv[i+1] - tv[i]) * (t - tv[i])
    end
    return nothing
end
