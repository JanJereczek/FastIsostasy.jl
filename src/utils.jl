#####################################################
# Unit conversion utils
#####################################################

function years2seconds(t::T) where {T<:AbstractFloat}
    return t * seconds_per_year
end

function seconds2years(t::T) where {T<:AbstractFloat}
    return t / seconds_per_year
end

function m_per_sec2mm_per_yr(dudt::T) where {T<:AbstractFloat}
    return dudt * 1e3 * seconds_per_year
end

#####################################################
# Array utils
#####################################################

"""

    matrify_vectorconstant(x, N)

Generate a vector of constant matrices from a vector of constants.
"""
function matrify_vectorconstant(x::Vector{T}, N::Int) where {T<:AbstractFloat}
    X = zeros(T, N, N, length(x))
    for i in eachindex(x)
        X[:, :, i] = matrify_constant(x[i], N)
    end
    return X
end

"""

    matrify_constant(x, N)

Generate a constant matrix from a constant.
"""
function matrify_constant(x::T, N::Int) where {T<:AbstractFloat}
    return fill(x, N, N)
end

#####################################################
# Domain and projection utils
#####################################################

"""

    get_r(x, y)

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:Real} = sqrt(x^2 + y^2)
get_r(X::Matrix{T}, Y::Matrix{T}) where {T<:Real} = get_r.(X, Y)

"""

    meshgrid(x, y)

Return a 2D meshgrid spanned by `x, y`.
"""
function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return one_y * x', (one_x * y')'
end

"""

    dist2angulardist()

"""
function dist2angulardist(
    dist::T,
) where {T<:Real}
    r_equator = 6.371e6
    return 2 * atan( dist / (2 * r_equator) )
end

"""

    sphericaldistance()

"""
function sphericaldistance(
    lat::T,
    lon::T;
    lat0::T = T(-pi / 2),   # default origin is south pole
    lon0::T = T(0),         # default origin is south pole
    R::T = T(6.371e6),      # Earth radius
) where {T<:Real}
    return R * acos( sin(lat) * sin(lat0) + cos(lat) * cos(lat0) * (lon - lon0) )
end

"""

    latlon2stereo()

Convert latitude-longitude coordinates to stereographically projected (x,y).
Reference: John P. Snyder (1987), p. 157, eq. (21-2), (21-3), (21-4).
"""
function latlon2stereo(
    lat::T,
    lon::T;
    lat0=T(-90.0),          # reference latitude of projection, FIXME: does not work properly with lat0 = -71° for now
    lon0=T(0.0),            # reference longitude of projection
    R::T = T(6.371e6),      # Earth radius
    k0::T = T(1.0),         # Scale factor
) where {T<:Real}
    lat, lon, lat0, lon0 = deg2rad.([lat, lon, lat0, lon0])
    k = 2*k0 / (1 + sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
    x = R * k * cos(lat) * sin(lon - lon0)
    y = R * k * (cos(lat0) * sin(lat) - sin(lat0) * cos(lat) * cos(lon-lon0))
    return k, x, y
end

function latlon2stereo(
    lat::AbstractMatrix{T},
    lon::AbstractMatrix{T};
    kwargs...,
) where {T<:Real}
    K, X, Y = copy(lat), copy(lat), copy(lat)
    for idx in CartesianIndices(lat)
        K[idx], X[idx], Y[idx] = latlon2stereo(lat[idx], lon[idx], kwargs...)
    end
    return K, X, Y
end

"""

    stereo2latlon()

Convert stereographic (x,y)-coordinates to latitude-longitude.
Reference: John P. Snyder (1987), p. 159, eq. (20-14), (20-15), (20-18), (21-15).
"""
function stereo2latlon(
    x::T,
    y::T;
    lat0=T(-90.0),          # reference latitude of projection, FIXME: does not work properly with lat0 = -71° for now
    lon0=T(0.0),            # reference longitude of projection
    R::T = T(6.371e6),      # Earth radius
    k0::T = T(1.0), 
) where {T<:Real}
    lat0, lon0 = deg2rad.([lat0, lon0])
    r = get_r(x, y) + 1e-12     # add small tolerance to avoid division by zero
    c = 2 * atan( r/(2*R*k0) )
    lat = asin( cos(c) * sin(lat0) + y/r * sin(c) * cos(lat0) )
    lon = lon0 + atan( x*sin(c), (r * cos(lat0) * cos(c) - y * sin(lat0) * sin(c)) )
    return rad2deg(lat), rad2deg(lon)
end

function stereo2latlon(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T};
    kwargs...,
) where {T<:Real}
    Lat, Lon = copy(x), copy(x)
    for idx in CartesianIndices(x)
        Lat[idx], Lon[idx] = stereo2latlon(x[idx], y[idx], kwargs...)
    end
    return Lat, Lon
end

"""

    init_domain(L, n)

Initialize a square computational domain with length `2*L` and `2^n` grid cells.
"""
function init_domain(
    L::T,
    n::Int;
    use_cuda=false::Bool
) where {T<:AbstractFloat}

    # Geometry
    Lx, Ly = L, L
    N = 2^n
    N2 = Int(floor(N/2))
    dx = T(2*Lx) / N
    dy = T(2*Ly) / N
    x = collect(-Lx+dx:dx:Lx)
    y = collect(-Ly+dy:dy:Ly)
    X, Y = meshgrid(x, y)
    R = get_r.(X, Y)

    Lat, Lon = stereo2latlon(x, y)

    Θ = dist2angulardist.(R)
    
    arraykernel = use_cuda ? CuArray : Array
    
    # Differential operators in Fourier space
    pseudodiff, harmonic, biharmonic = get_differential_fourier(L, N2)
    pseudodiff[1, 1] = mean([pseudodiff[1,2], pseudodiff[2,1]])
    pseudodiff, harmonic, biharmonic = kernelpromote(
            [pseudodiff, harmonic, biharmonic], arraykernel)
    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.

    return ComputationDomain(
        Lx, Ly, N, N2,
        dx, dy, x, y,
        X, Y, R, Θ,
        pseudodiff, harmonic, biharmonic,
        use_cuda, arraykernel
    )
end


#####################################################
# Math utils
#####################################################

function gauss_distr(
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    mu::Vector{T},
    sigma::Matrix{T}
) where {T<:AbstractFloat}
    k = length(mu)
    G = similar(X)
    invsigma = inv(sigma)
    invsqrtdetsigma = 1/sqrt(det(sigma))
    for i in axes(X,1), j in axes(X,2)
        G[i, j] = (2*π)^(-k/2) * invsqrtdetsigma * exp( 
            -0.5 * ([X[i,j], Y[i,j]] .- mu)' * invsigma * ([X[i,j], Y[i,j]] .- mu) )
    end
    return G
end

#####################################################
# Solid Earth parameters
#####################################################

litho_rigidity = 5e24               # (N*m)
litho_youngmodulus = 6.6e10         # (N/m^2)
litho_poissonratio = 0.5            # (1)
layers_density = [3.3e3]            # (kg/m^3)
layers_viscosity = [1e19, 1e21]     # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
layers_begin = [88e3, 400e3]
# 88 km: beginning of asthenosphere (Bueler 2007).
# 400 km: beginning of homogenous half-space (Ivins 2022, Fig 12).

"""

    init_multilayer_earth(
        Omega::ComputationDomain{T};
        litho_rigidity<:Union{Vector{T}, Vector{AbstractMatrix{T}}},
        layers_density::Vector{T},
        layers_viscosity<:Union{Vector{T}, Vector{AbstractMatrix{T}}},
        layers_begin::Vector{T},
    ) where {T<:AbstractFloat}

Return struct with solid-Earth parameters for mutliple channel layers and a halfspace.
"""
function init_multilayer_earth(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T};
    layers_begin::A = layers_begin,
    layers_density::Vector{T} = layers_density,
    layers_viscosity::B = layers_viscosity,
    litho_youngmodulus::C = litho_youngmodulus,
    litho_poissonratio::D = litho_poissonratio,
) where {
    T<:AbstractFloat,
    A<:Union{Vector{T}, Array{T, 3}},
    B<:Union{Vector{T}, Array{T, 3}},
    C<:Union{T, AbstractMatrix{T}},
    D<:Union{T, AbstractMatrix{T}},
}

    if layers_begin isa Vector
        layers_begin = matrify_vectorconstant(layers_begin, Omega.N)
    end
    litho_thickness = layers_begin[:, :, 1]
    litho_rigidity = get_rigidity.(
        litho_thickness,
        litho_youngmodulus,
        litho_poissonratio,
    )

    if layers_viscosity isa Vector
        layers_viscosity = matrify_vectorconstant(layers_viscosity, Omega.N)
    end

    layers_thickness = diff( layers_begin, dims=3 )
    # pseudodiff = kernelpromote(Omega.pseudodiff, Omega.arraykernel)
    effective_viscosity = get_effective_viscosity(
        Omega,
        layers_viscosity,
        layers_thickness,
        # pseudodiff,
    )

    # mean_density = get_matrix_mean_density(layers_thickness, layers_density)
    mean_density = fill(layers_density[1], Omega.N, Omega.N)

    litho_rigidity, effective_viscosity, mean_density = kernelpromote(
        [litho_rigidity, effective_viscosity, mean_density], Omega.arraykernel)

    return MultilayerEarth(
        c.g,
        mean(mean_density),
        effective_viscosity,
        litho_thickness,
        litho_rigidity,
        litho_poissonratio,
        layers_density,
        layers_viscosity,
        layers_begin,
    )

end

function get_rigidity(
    t::T,
    E::T,
    nu::T,
) where {T<:AbstractFloat}
    return (E * t^3) / (12 * (1 - nu^2))
end

function get_matrix_mean_density(
    layers_thickness::Array{T, 3},
    layers_density::Vector{T},
) where {T<:AbstractFloat}
    mean_density = zeros(T, size(layers_thickness)[1:2])
    for i in axes(layers_thickness, 1), j in axes(layers_thickness, 2)
        mean_density[i, j] = get_mean_density(layers_thickness[i, j, :], layers_density)
    end
    return mean_density
end

function get_mean_density(
    layers_thickness::Vector{T},
    layers_density::Vector{T},
) where {T<:AbstractFloat}
    return sum( (layers_thickness ./ (sum(layers_thickness)))' * layers_density )
end

function matrified_mean_gravity()
    fixed_mean_gravity = true
    if fixed_mean_gravity
        mean_gravity = c.g
    else
        # Earth acceleration over depth z.
        # Use the pole radius because GIA most important at poles.
        gr(r) = 4*π/3*c.G*c.rho_0* r - π*c.G*(c.rho_0 - c.rho_1) * r^2/c.r_pole
        gz(z) = gr(c.r_pole - z)
        mean_gravity = get_mean_gravity(layers_begin, layers_thickness, gz)
    end
end

function get_mean_gravity(
    layers_begin::Vector{T},
    layers_thickness::Vector{T},
    gz::Function,
) where {T<:AbstractFloat}
    layers_mean_gravity = 0.5 .*(gz.(layers_begin[1:end-1]) + gz.(layers_begin[2:end]))
    return (layers_thickness ./ (sum(layers_thickness)))' * layers_mean_gravity
end

"""

    get_effective_viscosity(
        layers_viscosity::Vector{AbstractMatrix{T}},
        layers_thickness::Vector{T},
        Omega::ComputationDomain{T},
    ) where {T<:AbstractFloat}

Compute equivalent viscosity for multilayer model by recursively applying
the formula for a halfspace and a channel from Lingle and Clark (1975).
"""
function get_effective_viscosity(
    Omega::ComputationDomain{T},
    layers_viscosity::Array{T, 3},
    layers_thickness::Array{T, 3},
    # pseudodiff::AbstractMatrix{T},
) where {T<:AbstractFloat}

    # Recursion has to start with half space = n-th layer:
    effective_viscosity = layers_viscosity[:, :, end]
    # p1, p2 = plan_fft(effective_viscosity), plan_ifft(effective_viscosity)
    for i in axes(layers_viscosity, 3)[1:end-1]
        channel_viscosity = layers_viscosity[:, :, end - i]
        channel_thickness = layers_thickness[:, :, end - i + 1]
        viscosity_ratio = get_viscosity_ratio(channel_viscosity, effective_viscosity)
        viscosity_scaling = three_layer_scaling(
            Omega,
            # pseudodiff,
            viscosity_ratio,
            channel_thickness,
        )
        copy!( 
            effective_viscosity,
            effective_viscosity .* viscosity_scaling,
        )
    end
    return effective_viscosity
end


"""

    get_viscosity_ratio(
        channel_viscosity::Matrix{T},
        halfspace_viscosity::Matrix{T},
    )

Return the viscosity ratio between channel and half-space as specified in
Bueler (2007) below equation 15.

"""
function get_viscosity_ratio(
    channel_viscosity::Matrix{T},
    halfspace_viscosity::Matrix{T},
) where {T<:AbstractFloat}
    return channel_viscosity ./ halfspace_viscosity
end

"""

    three_layer_scaling(
        Omega,
        kappa::T,
        visc_ratio::T,
        channel_thickness::T,
    )

Return the viscosity scaling for three-layer model as given in
Bueler (2007) below equation 15.

"""
function three_layer_scaling(
    # kappa::Matrix{T},
    Omega::ComputationDomain{T},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
) where {T<:AbstractFloat}

    # FIXME: What is kappa in that context???
    kappa = π / Omega.Lx
    C = cosh.(channel_thickness .* kappa)
    S = sinh.(channel_thickness .* kappa)

    num1 = 2 .* visc_ratio .* C .* S
    num2 = (1 .- visc_ratio .^ 2) .* channel_thickness .^ 2 .* kappa .^ 2
    num3 = visc_ratio .^ 2 .* S .^ 2 + C .^ 2

    denum1 = (visc_ratio .+ 1 ./ visc_ratio) .* C .* S
    denum2 = (visc_ratio .- 1 ./ visc_ratio) .* channel_thickness .* kappa
    denum3 = S .^ 2 + C .^ 2
    
    return (num1 + num2 + num3) ./ (denum1 + denum2 + denum3)
end

"""

    loginterp_viscosity(tvec, layers_viscosity, layers_thickness, pseudodiff)

Compute a log-interpolator of the equivalent viscosity from provided viscosity
fields `layers_viscosity` at time stamps `tvec`.
"""
function loginterp_viscosity(
    tvec::AbstractVector{T},
    layers_viscosity::Array{T, 4},
    layers_thickness::Array{T, 3},
    pseudodiff::AbstractMatrix,
) where {T<:AbstractFloat}
    n1, n2, n3, nt = size(layers_viscosity)
    log_eqviscosity = [fill(T(0.0), n1, n2) for k in 1:nt]

    [log_eqviscosity[k] .= log10.(get_effective_viscosity(
        layers_viscosity[:, :, :, k],
        layers_thickness,
        pseudodiff,
    )) for k in 1:nt]

    log_interp = linear_interpolation(tvec, log_eqviscosity)
    visc_interp(t) = 10 .^ log_interp(t)
    return visc_interp
end

"""

    hyperbolic_channel_coeffs(channel_thickness, kappa)

Return hyperbolic coefficients for equivalent viscosity computation.
"""
function hyperbolic_channel_coeffs(
    channel_thickness::T,
    kappa::T,
) where {T<:AbstractFloat}
    return cosh(channel_thickness * kappa), sinh(channel_thickness * kappa)
end

#####################################################
# Load response utils
#####################################################

"""

    get_greenintegrand_coeffs(T)

Return the load response coefficients with type `T` and listed in table A3 of
Farrell (1972).
"""
function get_greenintegrand_coeffs(T::Type)

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

    build_greenintegrand(x, y)

Compute the load response matrix of the solid Earth based on Green's function
(c.f. Farell 1972). In the Fourier space, this corresponds to a product which
is subsequently transformed back into the time domain.
Use pre-computed integration tools to accelerate computation.
"""
function build_greenintegrand(
    distance::Vector{T},
    greenintegrand_coeffs::Vector{T},
) where {T<:AbstractFloat}

    greenintegrand_interp = linear_interpolation(distance, greenintegrand_coeffs)
    compute_greenintegrand_entry_r(r::T) = compute_greenintegrand_entry(
        r,
        distance,
        greenintegrand_coeffs,
        greenintegrand_interp,
    )
    greenintegrand_function(x::T, y::T) = compute_greenintegrand_entry_r( get_r(x, y) )
    return greenintegrand_function
end

function compute_greenintegrand_entry(
    r::T,
    rm::Vector{T},
    greenintegrand_coeffs::Vector{T},
    interp_greenintegrand_::Interpolations.Extrapolation,
) where {T<:AbstractFloat}

    if r < 0.01
        return greenintegrand_coeffs[1] / ( rm[2] * T(1e12) )
    elseif r > rm[end]
        return T(0.0)
    else
        return interp_greenintegrand_(r) / ( r * T(1e12) )
    end
end

"""

    get_elasticgreen(Omega, quad_support, quad_coeffs)

Integrate load response over field by using 2D quadrature with specified
support points and associated coefficients.
"""
function get_elasticgreen(
    Omega::ComputationDomain{T},
    greenintegrand_function::Function,
    quad_support::Vector{T},
    quad_coeffs::Vector{T},
) where {T<:AbstractFloat}

    h = Omega.dx
    N = Omega.N
    elasticgreen = similar(Omega.X)

    for i = 1:N, j = 1:N
        p = i - Omega.N2 - 1
        q = j - Omega.N2 - 1
        elasticgreen[i, j] = quadrature2D(
            greenintegrand_function,
            quad_support,
            quad_coeffs,
            p*h,
            p*h+h,
            q*h,
            q*h+h,
        )
    end
    return elasticgreen
end

#####################################################
# Quadrature utils
#####################################################

"""

    get_quad_coeffs(T, n)

Return support points and associated coefficients with specified Type
for Gauss-Legendre quadrature.
"""
function get_quad_coeffs(T::Type, n::Int)
    x, w = gausslegendre( n )
    return T.(x), T.(w)
end


"""

    quadrature1D(f, n, x1, x2)

Compute 1D Gauss-Legendre quadrature of `f` between `x1` and `x2`
based on `n` support points.
"""
function quadrature1D(f::Function, n::Int, x1::T, x2::T) where {T<:AbstractFloat}
    x, w = get_quad_coeffs(T, n)
    m, p = get_normalized_lin_transform(x1, x2)
    sum = 0
    for i=1:n
        sum = sum + f(normalized_lin_transform(x[i], m, p)) * w[i] / m
    end
    return sum
end

"""

    quadrature2D(f, x, w, x1, x2, y1, y2)

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
    for i=1:n, j=1:n
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
function get_normalized_lin_transform(x1::T, x2::T) where {T<:AbstractFloat}
    x1_norm, x2_norm = T(-1), T(1)
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

"""

    normalized_lin_transform(y, m, p)

Apply normalized linear transformation with slope `m` and bias `p` on `y`.
"""
function normalized_lin_transform(y::T, m::T, p::T) where {T<:AbstractFloat}
    return (y-p)/m
end


#####################################################
# Kernel utils
#####################################################

function kernelpromote(X::M, arraykernel) where {M<:AbstractArray{T}} where {T<:Real}
    return arraykernel(X)
end

function kernelpromote(X::Vector{M}, arraykernel) where {M<:AbstractArray{T}} where {T<:Real}
    return [arraykernel(x) for x in X]
end

function convert2CuArray(X::Vector)
    return [CuArray(x) for x in X]
end

function convert2Array(X::Vector)
    return [Array(x) for x in X]
end

function copystructs2cpu(
    Omega::ComputationDomain{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T},
) where {T<:AbstractFloat}

    n = Int( round( log2(Omega.N) ) )
    Omega_cpu = init_domain(Omega.Lx, n, use_cuda = false)

    p_cpu = init_multilayer_earth(
        Omega_cpu,
        c;
        layers_begin = Array(p.layers_begin),
        layers_density = Array(p.layers_density),
        layers_viscosity = Array(p.layers_viscosity),
    )

    return Omega_cpu, p_cpu
end