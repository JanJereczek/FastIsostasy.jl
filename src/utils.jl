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
    distance, loadresponse_coeffs = get_loadresponse_coeffs(T)
    loadresponse_matrix, loadresponse_function = build_loadresponse_matrix(X, Y, distance, loadresponse_coeffs)
    return ComputationDomain(L, N, h, x, X, Y, loadresponse_matrix, loadresponse_function)
end

struct ComputationDomain{T<:AbstractFloat}
    L::T
    N::Int
    h::T
    x::Vector{T}
    X::Matrix{T}
    Y::Matrix{T}
    loadresponse_matrix::Matrix{T}
    loadresponse_function::Function
end

#####################################################
############## Load response matrix #################
#####################################################

function get_loadresponse_coeffs(T::Type)
    # Earth's radius
    a = 6.371e6

    # Angles of table A3 of
    # Deformation of the Earth by surface Loads, Farrell 1972
    θ = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1,
         0.16,   0.2,   0.25, 0.3,  0.4,  0.5,  0.6,  0.8,  1.0,
         1.2,    1.6,   2.0,  2.5,  3.0,  4.0,  5.0,  6.0,  7.0,
         8.0,    9.0,   10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0,
         50.0,   60.0,  70.0, 80.0, 90.0]


    # Column 1 (converted by some factor) of table A3 of
    # Deformation of the Earth by surface Loads, Farrell 1972
    rm = [ 0.0,    0.011,  0.111,  1.112,  2.224,  3.336,  4.448,  6.672,
           8.896,  11.12,  17.79,  22.24,  27.80,  33.36,  44.48,  55.60, 
           66.72,  88.96,  111.2,  133.4,  177.9,  222.4,  278.0,  333.6,
           444.8,  556.0,  667.2,  778.4,  889.6,  1001.0, 1112.0, 1334.0,
           1779.0, 2224.0, 2780.0, 3336.0, 4448.0, 5560.0, 6672.0,
           7784.0, 8896.0, 10008.0] .* 1e3
    # converted to meters
    # GE /(10^12 rm) is vertical displacement in meters (applied load is 1kg)

    # Column 2 of table A3 of
    # Deformation of the Earth by surface Loads, Farrell 1972
    GE = [ -33.6488, -33.64, -33.56, -32.75, -31.86, -30.98, -30.12, -28.44, -26.87, -25.41,
           -21.80, -20.02, -18.36, -17.18, -15.71, -14.91, -14.41, -13.69, -13.01,
           -12.31, -10.95, -9.757, -8.519, -7.533, -6.131, -5.237, -4.660, -4.272,
           -3.999, -3.798, -3.640, -3.392, -2.999, -2.619, -2.103, -1.530, -0.292,
            0.848,  1.676,  2.083,  2.057,  1.643];
    return T.(rm), T.(GE)
end

"""

    get_r(x, y)

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:AbstractFloat} = LinearAlgebra.norm([x, y])

"""

    build_loadresponse_matrix(x, y)

Compute the load response matrix of the solid Earth based on Green's function
(c.f. Farell 1972). In the Fourier space, this corresponds to a product which
is subsequently transformed back into the time domain.
Use pre-computed integration tools to accelerate computation.
"""
function build_loadresponse_matrix(
    X::Matrix{T},
    Y::Matrix{T},
    distance::Vector{T},
    loadresponse_coeffs::Vector{T},
) where {T<:AbstractFloat}

    loadresponse_interp = linear_interpolation(distance, loadresponse_coeffs)
    compute_loadresponse_entry_r(r::T) = compute_loadresponse_entry(
        r,
        distance,
        loadresponse_coeffs,
        loadresponse_interp,
    )
    loadresponse_matrix = compute_loadresponse_entry_r.( get_r.(X, Y) )
    loadresponse_function(x::T, y::T) = compute_loadresponse_entry_r( get_r(x, y) )
    loadresponse_matrix = loadresponse_function.(X, Y)
    
    return loadresponse_matrix, loadresponse_function
end

function compute_loadresponse_entry(
    r::T,
    rm::Vector{T},
    loadresponse_coeffs::Vector{T},
    interp_loadresponse_::Interpolations.Extrapolation,
) where {T<:AbstractFloat}

    if r < 0.01
        return loadresponse_coeffs[1] / ( rm[2] * T(1e12) )
    elseif r > rm[end]
        return T(0.0)
    else
        return interp_loadresponse_(r) / ( r * T(1e12) )
    end
end

#####################################################
############# Quadrature computation ################
#####################################################

function get_integrated_loadresponse(
    Omega::ComputationDomain,
    quad_support::Vector{T},
    quad_coeffs::Vector{T},
) where {T<:AbstractFloat}

    h = Omega.h
    N = Omega.N
    integrated_loadresponse = similar(Omega.X)
    Nx2 = Int(N/2)
    Ny2 = Int(N/2)

    @inline for i = 1:N, j = 1:N
        p = i - Nx2 - 1
        q = j - Ny2 - 1
        integrated_loadresponse[i, j] = Quad2D(
            Omega.loadresponse_function,
            quad_support,
            quad_coeffs,
            p*h,
            p*h+h,
            q*h,
            q*h+h,
        )
    end
    return integrated_loadresponse
end

function get_quad_coeffs(T::Type, n::Int)
    x, w = gausslegendre( n )
    return T.(x), T.(w)
end

function Quad2D(
    f::Function,
    x::Vector{T},
    w::Vector{T},
    x1::T,
    x2::T,
    y1::T,
    y2::T,
) where {T<:AbstractFloat}

    n = length(x)
    mx, px = get_lin_transform_2_norm(x1, x2)
    my, py = get_lin_transform_2_norm(y1, y2)
    sum = T(0)
    @inline for i=1:n, j=1:n
        sum = sum + f(
            lin_transform_2_norm(x[i], mx, px),
            lin_transform_2_norm(x[j], my, py),
        ) * w[i] * w[j] / mx / my
    end
    return sum
end

function Quad1D(f::Function, n::Int, x1::T, x2::T) where {T<:AbstractFloat}
    x, w = gausslegendre( n )
    m, p = get_lin_transform_2_norm(x1, x2)
    sum = 0
    for i=1:n
        sum = sum + f(lin_transform_2_norm(x[i], m, p)) * w[i] / m
    end
    return sum
end

function get_lin_transform_2_norm(x1::T, x2::T) where {T<:AbstractFloat}
    x1_norm, x2_norm = T(-1), T(1)
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

function lin_transform_2_norm(y::T, m::T, p::T) where {T<:AbstractFloat}
    return (y-p)/m
end