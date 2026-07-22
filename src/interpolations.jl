"""
$(TYPEDSIGNATURES)

Define the time interpolation of a scalar variable.

# Fields
- `t`: a vector of time points at which the variable is defined.
- `y`: a vector of variable values corresponding to `t`.
- `flat_bc`: a boolean indicating whether to use flat boundary conditions
"""
struct TimeInterpolation0D{T<:AbstractFloat}
    t::Vector{T}
    y::Vector{T}
    flat_bc::Bool
end

function TimeInterpolation0D(t, y; flat_bc = false)
    @assert length(t) == length(y)
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
    @assert length(t) == length(X)
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
    if t_out < minimum(itp.t)
        if itp.flat_bc
            return itp.y[1]
        else
            throw(ErrorException("TimeInterpolation0D out of range."))
        end
    elseif t_out > maximum(itp.t)
        if itp.flat_bc
            return itp.y[end]
        else
            throw(ErrorException("TimeInterpolation0D out of range."))
        end
    elseif t_out in itp.t
        i = searchsortedfirst(itp.t, t_out)
        return itp.y[i]
    else
        i = searchsortedfirst(itp.t, t_out) - 1
        return itp.y[i] +
               (itp.y[i+1] - itp.y[i]) / (itp.t[i+1] - itp.t[i]) * (t_out - itp.t[i])
    end
end

"""
$(TYPEDSIGNATURES)

Interpolate a time-dependent field (in-place) at a given time `t_out` using the interpolation object `itp`.
"""
function interpolate!(X_out::M, t::T, ti::TimeInterpolation2D{T,M}) where {T,M}
    if t < minimum(ti.t)
        if ti.flat_bc
            X_out .= ti.X[1]
        else
            throw(ErrorException("TimeInterpolation2D out of range."))
        end
    elseif t > maximum(ti.t)
        if ti.flat_bc
            X_out .= ti.X[end]
        else
            throw(ErrorException("TimeInterpolation2D out of range."))
        end
    elseif t in ti.t
        i = searchsortedfirst(ti.t, t)
        X_out .= ti.X[i]
    else
        i = searchsortedfirst(ti.t, t) - 1
        @. X_out =
            ti.X[i] + (ti.X[i+1] - ti.X[i]) / (ti.t[i+1] - ti.t[i]) * (t - ti.t[i])
    end
    return nothing
end
