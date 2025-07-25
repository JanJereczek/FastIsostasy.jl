struct TimeInterpolation0D{T<:AbstractFloat}
    t::Vector{T}
    y::Vector{T}
    flat_bc::Bool
end

function TimeInterpolation0D(t, y; flat_bc = false)
    @assert length(t) == length(y)
    return TimeInterpolation0D(t, y, flat_bc)
end

mutable struct TimeInterpolation2D{T<:AbstractFloat, M<:KernelMatrix{T}}
    t::Vector{T}
    X::Vector{M}
    flat_bc::Bool
end

function TimeInterpolation2D(t, X; flat_bc = false)
    @assert length(t) == length(X)
    return TimeInterpolation2D(t, X, flat_bc)
end


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
        return itp.y[i] + (itp.y[i+1] - itp.y[i]) /
            (itp.t[i+1] - itp.t[i]) * (t_out - itp.t[i])
    end
end

function interpolate!(X_out::M, t::T, ti::TimeInterpolation2D{T, M}) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
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
        @. X_out = ti.X[i] + (ti.X[i+1] - ti.X[i]) /
            (ti.t[i+1] - ti.t[i]) * (t - ti.t[i])
    end
    return nothing
end