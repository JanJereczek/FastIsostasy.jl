"""

    meshgrid(x::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}

Return a 2D meshgrid spanned by `x, y`.
"""
@inline function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return one_y * x', (one_x * y')'
end

"""

    init_domain(L::AbstractFloat, n::Int)

Initialize a square computational domain with length `2*L` and `2^n+1` grid cells.
"""
@inline function init_domain(L::T, n::Int) where {T<:AbstractFloat}
    N = 2^n
#     N = 2^n+1
    N2 = Int(floor(N/2))
    h = T(2*L) / N
    x = collect(-L+h:h:L)
#     x = collect(-L+h/2:h:L-h/2)
    X, Y = meshgrid(x, x)
    distance, loadresponse_coeffs = get_loadresponse_coeffs(T)
    loadresponse_matrix, loadresponse_function = build_loadresponse_matrix(X, Y, distance, loadresponse_coeffs)
    pseudodiff_coeffs, biharmonic_coeffs = get_differential_fourier(L, N2)

    return ComputationDomain(
        L,
        N,
        N2,
        h,
        x,
        X,
        Y,
        loadresponse_matrix,
        loadresponse_function,
        pseudodiff_coeffs,
        biharmonic_coeffs,
    )
end

struct ComputationDomain{T<:AbstractFloat}
    L::T
    N::Int
    N2::Int
    h::T
    x::Vector{T}
    X::AbstractMatrix{T}
    Y::AbstractMatrix{T}
    loadresponse_matrix::AbstractMatrix{T}
    loadresponse_function::Function
    pseudodiff_coeffs::AbstractMatrix{T}
    biharmonic_coeffs::AbstractMatrix{T}
end

"""
    get_differential_fourier(
        L::T,
        N2::Int,
    )

Compute the matrices capturing the differential operators in the fourier space.
"""
@inline function get_differential_fourier(
    L::T,
    N2::Int,
) where {T<:Real}
    mu = T(π / L)
    raw_coeffs = mu .* T.( vcat(0:N2, N2-1:-1:1) )
    x_coeffs, y_coeffs = raw_coeffs, raw_coeffs
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    laplacian_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(laplacian_coeffs)
    biharmonic_coeffs = laplacian_coeffs .^ 2
    return pseudodiff_coeffs, biharmonic_coeffs
end
#####################################################
############### Physical constants ##################
#####################################################

g = 9.81                                    # m/s^2
seconds_per_year = 60 * 60 * 24 * 365       # s
rho_ice = 0.910e3                           # kg/m^3

"""
    init_physical_constants(T::Type)

Return struct containing physical constants.
"""
@inline function init_physical_constants(T::Type)
    return PhysicalConstants(T(g), T(seconds_per_year), T(rho_ice))
end

struct PhysicalConstants{T<:AbstractFloat}
    g::T
    seconds_per_year::T
    rho_ice::T
end

#####################################################
############# Solid Earth parameters ################
#####################################################

mantle_density = 3.3e3              # kg/m^3
lithosphere_rigidity = 5e24         # N*m
halfspace_viscosity = 1e21          # Pa*s (Ivins 2022, Fig 12 WAIS)
channel_viscosity = 1e19            # Pa*s (Ivins 2022, Fig 10 WAIS)
channel_begin = 88e3                # 88 km: beginning of asthenosphere (Bueler 2007).
halfspace_begin = 400e3             # 400 km: beginning of homogenous half-space (Ivins 2022, Fig 12).

"""

    init_solidearth_params(
        T::Type,
        Omega::ComputationDomain;
        lithosphere_rigidity,
        mantle_density,
        channel_viscosity,
        halfspace_viscosity,
        channel_begin,
        halfspace_begin,
    )

Return struct containing solid-Earth parameters.
"""
@inline function init_solidearth_params(
    T::Type,
    Omega::ComputationDomain;
    lithosphere_rigidity = fill(T(lithosphere_rigidity), Omega.N, Omega.N),
    mantle_density = fill(T(mantle_density), Omega.N, Omega.N),
    channel_viscosity = fill(T(channel_viscosity), Omega.N, Omega.N),
    halfspace_viscosity = fill(T(halfspace_viscosity), Omega.N, Omega.N),
    channel_begin = fill(T(channel_begin), Omega.N, Omega.N),
    halfspace_begin = fill(T(halfspace_begin), Omega.N, Omega.N),
)

    channel_thickness = halfspace_begin - channel_begin
    viscosity_ratio = get_viscosity_ratio(channel_viscosity, halfspace_viscosity)
    viscosity_scaling = three_layer_scaling(
        Omega.pseudodiff_coeffs,       # κ
        viscosity_ratio,
        channel_thickness,
    )

    return SolidEarthParams(
        lithosphere_rigidity,
        mantle_density,
        channel_viscosity,
        halfspace_viscosity,
        viscosity_ratio,
        viscosity_scaling,
        channel_begin,
        halfspace_begin,
        channel_thickness,
    )
end

struct SolidEarthParams{T<:AbstractFloat}
    lithosphere_rigidity::AbstractMatrix{T}
    mantle_density::AbstractMatrix{T}
    channel_viscosity::AbstractMatrix{T}
    halfspace_viscosity::AbstractMatrix{T}
    viscosity_ratio::AbstractMatrix{T}
    viscosity_scaling::AbstractMatrix{T}
    channel_begin::AbstractMatrix{T}
    halfspace_begin::AbstractMatrix{T}
    channel_thickness::AbstractMatrix{T}
end

"""

    get_viscosity_ratio(
        channel_viscosity::Matrix{T},
        halfspace_viscosity::Matrix{T},
    )

Return the viscosity ratio between channel and half-space as specified in
Bueler (2007) below equation 15.

"""
@inline function get_viscosity_ratio(
    channel_viscosity::Matrix{T},
    halfspace_viscosity::Matrix{T},
) where {T<:AbstractFloat}
    return channel_viscosity ./ halfspace_viscosity
end

"""

    three_layer_scaling(
        kappa::T,
        visc_ratio::T,
        Tc::T,
    )

Return the viscosity scaling for three-layer model as given in
Bueler (2007) below equation 15.

"""
@inline function three_layer_scaling(
    kappa::Matrix{T},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
) where {T<:AbstractFloat}

    visc_scaling = zeros(T, size(kappa)...)
    for i in axes(kappa, 1), j in axes(kappa, 2)

        k = π / 2e6 # kappa[i, j]
        vr = visc_ratio[i, j]
        Tc = channel_thickness[i, j]     # Lingle-Clark: in km

        C, S = hyperbolic_channel_coeffs(Tc, k)
        
        num1 = 2 * vr * C * S
        num2 = (1 - vr ^ 2) * Tc^2 * k ^ 2
        num3 = vr ^ 2 * S ^ 2 + C ^ 2

        denum1 = (vr + 1/vr) * C * S
        denum2 = (vr - 1/vr) * Tc * k
        denum3 = S^2 + C^2
        
        visc_scaling[i, j] = (num1 + num2 + num3) / (denum1 + denum2 + denum3)
    end
    return visc_scaling
end

@inline function hyperbolic_channel_coeffs(
    Tc::T,
    kappa::T,
) where {T<:AbstractFloat}
    return cosh(Tc * kappa), sinh(Tc * kappa)
end

#####################################################
############## Geometric utilities ##################
#####################################################

"""

    get_r(x, y)

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:Real} = LinearAlgebra.norm([x, y])

#####################################################
############## Load response matrix #################
#####################################################

"""

    get_loadresponse_coeffs(T)

Return the load response coefficients with type `T` and listed in table A3 of
Farrell (1972).
"""
function get_loadresponse_coeffs(T::Type)
    # Earth's radius
    a = 6.371e6 # equator
    a = 6.357e6 # pole

    # Angles of table A3 of
    # Deformation of the Earth by surface Loads, Farrell 1972
    θ = [0.0, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1,
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

    build_loadresponse_matrix(x, y)

Compute the load response matrix of the solid Earth based on Green's function
(c.f. Farell 1972). In the Fourier space, this corresponds to a product which
is subsequently transformed back into the time domain.
Use pre-computed integration tools to accelerate computation.
"""
@inline function build_loadresponse_matrix(
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
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

@inline function compute_loadresponse_entry(
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

"""

    get_integrated_loadresponse(Omega, quad_support, quad_coeffs)

Integrate load response over field by using 2D quadrature with specified
support points and associated coefficients.
"""
@inline function get_integrated_loadresponse(
    Omega::ComputationDomain,
    quad_support::Vector{T},
    quad_coeffs::Vector{T},
) where {T<:AbstractFloat}

    h = Omega.h
    N = Omega.N
    integrated_loadresponse = similar(Omega.X)

    @inline for i = 1:N, j = 1:N
        p = i - Omega.N2 - 1
        q = j - Omega.N2 - 1
        integrated_loadresponse[i, j] = quadrature2D(
            Omega.loadresponse_function,
            quad_support,
            quad_coeffs,
            p*h,
            p*h+h,
            q*h,
            q*h+h,
            # p*h-h/2,
            # p*h+h/2,
            # q*h-h/2,
            # q*h+h/2,
        )
    end
    return integrated_loadresponse
end

"""

    get_quad_coeffs(T, n)

Return support points and associated coefficients with specified Type
for Gauss-Legendre quadrature.
"""
@inline function get_quad_coeffs(T::Type, n::Int)
    x, w = gausslegendre( n )
    return T.(x), T.(w)
end


"""

    quadrature1D(f, n, x1, x2)

Compute 1D Gauss-Legendre quadrature of `f` between `x1` and `x2`
based on `n` support points.
"""
@inline function quadrature1D(f::Function, n::Int, x1::T, x2::T) where {T<:AbstractFloat}
    x, w = get_quad_coeffs(T, n)
    m, p = get_normalized_lin_transform(x1, x2)
    sum = 0
    for i=1:n
        sum = sum + f(normalized_lin_transform(x[i], m, p)) * w[i] / m
    end
    return sum
end

"""

    quadrature2D(
        f::Function,
        x::Vector{T},
        w::Vector{T},
        x1::T,
        x2::T,
        y1::T,
        y2::T,
    )

Return the integration of `f` over [`x1, x2`] x [`y1, y2`] with `x, w` some pre-computed
support points and coefficients of the Gauss-Legendre quadrature.
"""
function quadrature2D(
    f::Function,
    x::Vector{T},
    w::Vector{T},
    x1::T,
    x2::T,
    y1::T,
    y2::T,
) where {T<:AbstractFloat}

    n = length(x)
    mx, px = get_normalized_lin_transform(x1, x2)
    my, py = get_normalized_lin_transform(y1, y2)
    sum = T(0)
    @inline for i=1:n, j=1:n
        sum = sum + f(
            normalized_lin_transform(x[i], mx, px),
            normalized_lin_transform(x[j], my, py),
        ) * w[i] * w[j] / mx / my
    end
    return sum
end

"""

    get_normalized_lin_transform(x1, x2)

Return parameters of linear function mapping `x1, x2` onto `-1, 1`.
"""
@inline function get_normalized_lin_transform(x1::T, x2::T) where {T<:AbstractFloat}
    x1_norm, x2_norm = T(-1), T(1)
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

"""

    normalized_lin_transform(y, m, p)

Apply normalized linear transformation with slope `m` and bias `p` on `y`.
"""
@inline function normalized_lin_transform(y::T, m::T, p::T) where {T<:AbstractFloat}
    return (y-p)/m
end