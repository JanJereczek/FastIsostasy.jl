struct TimeInterpolation0D{T<:AbstractFloat}
    t::Vector{T}
    X::Vector{T}
end

mutable struct TimeInterpolation2D{T<:AbstractFloat, M<:KernelMatrix{T}}
    t::Vector{T}
    X::Vector{M}
    tdiff::Vector{T}
    i::Int
    flat_bc::Bool
end

# mutable struct TimeInterpolation3D{T<:AbstractFloat, M<:KernelMatrix{T}}
#     t::Vector{T}
#     X::AbstractArray{T, 3}
#     tdiff::Vector{T}
#     i::Int
# end

# struct TimeInterpolation3D

function TimeInterpolation2D(t, X; flat_bc = false)
    tdiff = similar(t)
    i = 0
    return TimeInterpolation2D(t, X, tdiff, i, flat_bc)
end

# function piecewise_constant_interpolate!(X_out::M, t::T, ti::TimeInterpolation2D{T, M}) where
#     {T<:AbstractFloat, M<:KernelMatrix{T}}
#     ti.tdiff .= ti.t .- t
#     ti.i = argmin(abs.(ti.tdiff))
#     X_out .= ti.X[ti.i]
#     return nothing
# end

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