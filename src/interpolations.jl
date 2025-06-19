struct Interpolation0D{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    flat_bc::Bool
end

function Interpolation0D(x, y; flat_bc = false)
    if length(x) != length(y)
        throw(ErrorException("Interpolation0D: t and x must have the same length."))
    end
    return Interpolation0D(x, y, flat_bc)
end

function (itp::Interpolation0D)(x_out)
    if x_out < minimum(itp.x)
        if itp.flat_bc
            return itp.y[1]
        else
            throw(ErrorException("Interpolation0D out of range."))
        end
    elseif x_out > maximum(itp.x)
        if itp.flat_bc
            return itp.y[end]
        else
            throw(ErrorException("Interpolation0D out of range."))
        end
    elseif x_out in itp.x
        i = searchsortedfirst(itp.x, x_out)
        return itp.y[i]
    else
        i = searchsortedfirst(itp.x, x_out) - 1
        return itp.y[i] + (itp.y[i+1] - itp.y[i]) /
            (itp.x[i+1] - itp.x[i]) * (x_out - itp.x[i])
    end
end

mutable struct TimeInterpolation2D{T<:AbstractFloat, M<:KernelMatrix{T}}
    t::Vector{T}
    X::Vector{M}
    tdiff::Vector{T}
    i::Int
    flat_bc::Bool
end

function TimeInterpolation2D(t, X; flat_bc = false)
    tdiff = similar(t)
    i = 0
    return TimeInterpolation2D(t, X, tdiff, i, flat_bc)
end

function piecewise_linear_interpolate!(X_out::M, t::T, ti::TimeInterpolation2D{T, M}) where
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
        ti.i = searchsortedfirst(ti.t, t)
        X_out .= ti.X[ti.i]
    else
        ti.i = searchsortedfirst(ti.t, t) - 1
        X_out .= ti.X[ti.i] .+ (ti.X[ti.i+1] .- ti.X[ti.i]) ./
            (ti.t[ti.i+1] - ti.t[ti.i]) .* (t - ti.t[ti.i])
    end
    return nothing
end