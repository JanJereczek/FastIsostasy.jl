"""

    meshgrid(x::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}

Return a 2D meshgrid spanned by `x, y`.
"""
function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return one_y * x', reverse((one_x * y')', dims=1)
end

"""

    init_domain(L::AbstractFloat, N::Int)

Initialize a square computational domain with length `2*L` and `N^2` grid cells.
"""
function init_domain(L::T, N::Int) where {T<:AbstractFloat}
    h = T(2*L) / N
    x = collect(-L+h/2:h:L-h/2)
    X, Y = meshgrid(x, x)
    return DomainParams(L, N, h, x, X, Y)
end

struct DomainParams{T<:AbstractFloat}
    L::T
    N::Int
    h::T
    x::Vector{T}
    X::Matrix{T}
    Y::Matrix{T}
end