#####################################################
# Unit conversion utils
#####################################################

global SECONDS_PER_YEAR = 60^2 * 24 * 365.25

"""

    years2seconds(t::Real)

Convert input time `t` from years to seconds.
"""
function years2seconds(t::Real)
    return t * SECONDS_PER_YEAR
end

"""

    seconds2years(t::Real)

Convert input time `t` from seconds to years.
"""
function seconds2years(t::Real)
    return t / SECONDS_PER_YEAR
end

"""

    m_per_sec2mm_per_yr(dudt::Real)

Convert displacement rate `dudt` from \$ m \\, s^{-1} \$ to \$ mm \\, \\mathrm{yr}^{-1} \$.
"""
function m_per_sec2mm_per_yr(dudt::Real)
    return dudt * 1e3 * SECONDS_PER_YEAR
end

#####################################################
# Array utils
#####################################################

"""

    matrify_vectorconstant(x, N)

Generate a vector of constant matrices from a vector of constants.
"""
function matrify_vectorconstant(x::Vector{<:Real}, N::Int)
    return matrify_vectorconstant(x, N, N)
end

function matrify_vectorconstant(x::Vector{T}, Nx::Int, Ny::Int) where {T<:Real}
    X = zeros(T, Ny, Nx, length(x))
    @inbounds for i in eachindex(x)
        X[:, :, i] = matrify_constant(x[i], Nx, Ny)
    end
    return X
end

matrify_constant(x::Real, Nx::Int, Ny::Int) = fill(x, Ny, Nx)

#####################################################
# Domain and projection utils
#####################################################

"""

    get_r(x::T, y::T) where {T<:Real}

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:Real} = sqrt(x^2 + y^2)

"""

    meshgrid(x, y)

Return a 2D meshgrid spanned by `x, y`.
"""
function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return one_y * x', Matrix((one_x * y')')
end

"""

    dist2angulardist(r::Real)

Convert Euclidean to angular distance along great circle.
"""
function dist2angulardist(r::Real)
    R = 6.371e6     # radius at equator
    return 2 * atan( r / (2 * R) )
end

"""

    scalefactor(lat::T, lon::T, lat0::T, lon0::T) where {T<:Real}

Compute scaling factor of stereographic projection for a given latitude `lat`
longitude `lon`, reference latitude `lat0` and reference longitude `lon0`.
Optionally one can provide `lat::AbstractMatrix` and `lon::AbstractMatrix`
if the scale factor is to be computed for the whole domain.
Note: angles must be provided in radians!
Reference: John P. Snyder (1987), p. 157, eq. (21-4).
"""
function scalefactor(lat::T, lon::T, lat0::T, lon0::T; k0::T = T(1)) where {T<:Real}
    return 2*k0 / (1 + sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
end

function scalefactor(lat::XMatrix, lon::XMatrix, lat0::T, lon0::T;
    kwargs... ) where {T<:Real}
    K = similar(lat)
    @inbounds for idx in CartesianIndices(lat)
        K[idx] = scalefactor(lat[idx], lon[idx], lat0, lon0; kwargs...)
    end
    return K
end

"""

    latlon2stereo(lat, lon, lat0, lon0)

Compute stereographic projection (x,y) for a given latitude `lat`
longitude `lon`, reference latitude `lat0` and reference longitude `lon0`.
Optionally one can provide `lat::AbstractMatrix` and `lon::AbstractMatrix`
if the projection is to be computed for the whole domain.
Note: angles must be provided in degrees!
Reference: John P. Snyder (1987), p. 157, eq. (21-2), (21-3), (21-4).
"""
function latlon2stereo(lat::T, lon::T, lat0::T, lon0::T;
    R::T = T(6.371e6), kwargs...) where {T<:Real}
    lat, lon, lat0, lon0 = deg2rad.([lat, lon, lat0, lon0])
    k = scalefactor(lat, lon, lat0, lon0; kwargs...)
    x = R * k * cos(lat) * sin(lon - lon0)
    y = R * k * (cos(lat0) * sin(lat) - sin(lat0) * cos(lat) * cos(lon-lon0))
    return k, x, y
end

function latlon2stereo(
    lat::XMatrix,
    lon::XMatrix,
    lat0::T,
    lon0::T;
    kwargs...,
) where {T<:Real}
    K, X, Y = similar(lat), similar(lat), similar(lat)
    @inbounds for idx in CartesianIndices(lat)
        K[idx], X[idx], Y[idx] = latlon2stereo(lat[idx], lon[idx], lat0, lon0; kwargs...)
    end
    return K, X, Y
end

"""

    stereo2latlon(x, y, lat0, lon0)

Compute the inverse stereographic projection `(lat, lon)` based on Cartesian coordinates
`(x,y)` and for a given reference latitude `lat0` and reference longitude `lon0`.
Optionally one can provide `x::AbstractMatrix` and `y::AbstractMatrix`
if the projection is to be computed for the whole domain.
Note: angles must be  para elloprovided in degrees!

Convert stereographic (x,y)-coordinates to latitude-longitude.
Reference: John P. Snyder (1987), p. 159, eq. (20-14), (20-15), (20-18), (21-15).
"""
function stereo2latlon(x::T, y::T, lat0::T, lon0::T;
    R::T = T(6.371e6), k0::T = T(1)) where {T<:Real}
    lat0, lon0 = deg2rad.([lat0, lon0])
    r = get_r(x, y) + 1e-20     # add small tolerance to avoid division by zero
    c = 2 * atan( r/(2*R*k0) )
    lat = asin( cos(c) * sin(lat0) + y/r * sin(c) * cos(lat0) )
    lon = lon0 + atan( x*sin(c), (r * cos(lat0) * cos(c) - y * sin(lat0) * sin(c)) )
    return rad2deg(lat), rad2deg(lon)
end

function stereo2latlon(x::XMatrix, y::XMatrix, lat0::T, lon0::T;
    kwargs...) where {T<:Real}
    Lat, Lon = copy(x), copy(x)
    @inbounds for idx in CartesianIndices(x)
        Lat[idx], Lon[idx] = stereo2latlon(x[idx], y[idx], lat0, lon0; kwargs...)
    end
    return Lat, Lon
end


#####################################################
# Math utils
#####################################################

"""

    gauss_distr(X::XMatrix, Y::XMatrix, mu::Vector{<:Real}, sigma::Matrix{<:Real})

Compute `Z = f(X,Y)` with `f` a Gaussian function parametrized by mean
`mu` and covariance `sigma`.
"""
function gauss_distr(X::XMatrix, Y::XMatrix,
    mu::Vector{<:Real}, sigma::Matrix{<:Real})
    k = length(mu)
    G = similar(X)
    invsigma = inv(sigma)
    invsqrtdetsigma = 1/sqrt(det(sigma))
    @inbounds for i in axes(X,1), j in axes(X,2)
        G[i, j] = (2*π)^(-k/2) * invsqrtdetsigma * exp( 
            -0.5 * ([X[i,j], Y[i,j]] .- mu)' * invsigma * ([X[i,j], Y[i,j]] .- mu) )
    end
    return G
end

"""

    get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}

Compute rigidity `D` based on thickness `t`, Young modulus `E` and Poisson ration `nu`.
"""
function get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}
    return (E * t^3) / (12 * (1 - nu^2))
end

"""

    get_effective_viscosity(
        layer_viscosities::Vector{XMatrix},
        layers_thickness::Vector{T},
        Omega::ComputationDomain{T},
    ) where {T<:AbstractFloat}

Compute equivalent viscosity for multilayer model by recursively applying
the formula for a halfspace and a channel from Lingle and Clark (1975).
"""
function get_effective_viscosity(
    Omega::ComputationDomain{T},
    layer_viscosities::Array{T, 3},
    layers_thickness::Array{T, 3},
    # pseudodiff::XMatrix,
) where {T<:AbstractFloat}

    # Recursion has to start with half space = n-th layer:
    effective_viscosity = layer_viscosities[:, :, end]
    # p1, p2 = plan_fft(effective_viscosity), plan_ifft(effective_viscosity)
    @inbounds for i in axes(layer_viscosities, 3)[1:end-1]
        channel_viscosity = layer_viscosities[:, :, end - i]
        channel_thickness = layers_thickness[:, :, end - i + 1]
        viscosity_ratio = channel_viscosity ./ effective_viscosity
        viscosity_scaling = three_layer_scaling(
            Omega,
            viscosity_ratio,
            channel_thickness,
        )
        effective_viscosity .*= viscosity_scaling
    end
    return effective_viscosity
end

"""

    three_layer_scaling(Omega::ComputationDomain, kappa::T, visc_ratio::T,
        channel_thickness::T)

Return the viscosity scaling for a three-layer model and based on a the wave
number `kappa`, the `visc_ratio` and the `channel_thickness`.
Reference: Bueler et al. 2007, below equation 15.
"""
function three_layer_scaling(
    # kappa::Matrix{T},
    Omega::ComputationDomain{T},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
) where {T<:AbstractFloat}

    # kappa is the wavenumber of the harmonic load. (see Cathles 1975, p.43)
    # we assume this is related to the size of the domain!
    kappa = π / Omega.Wx

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

    loginterp_viscosity(tvec, layer_viscosities, layers_thickness, pseudodiff)

Compute a log-interpolator of the equivalent viscosity from provided viscosity
fields `layer_viscosities` at time stamps `tvec`.
"""
function loginterp_viscosity(
    tvec::AbstractVector{T},
    layer_viscosities::Array{T, 4},
    layers_thickness::Array{T, 3},
    pseudodiff::XMatrix,
) where {T<:AbstractFloat}
    n1, n2, n3, nt = size(layer_viscosities)
    log_eqviscosity = [fill(T(0.0), n1, n2) for k in 1:nt]

    [log_eqviscosity[k] .= log10.(get_effective_viscosity(
        layer_viscosities[:, :, :, k],
        layers_thickness,
        pseudodiff,
    )) for k in 1:nt]

    log_interp = linear_interpolation(tvec, log_eqviscosity)
    visc_interp(t) = 10 .^ log_interp(t)
    return visc_interp
end

#####################################################
# Load response utils
#####################################################

"""

    get_greenintegrand_coeffs(T)

Return the load response coefficients with type `T`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function get_greenintegrand_coeffs(T::Type)

    # Angles
    θ = [0.0, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1,
         0.16,   0.2,   0.25, 0.3,  0.4,  0.5,  0.6,  0.8,  1.0,
         1.2,    1.6,   2.0,  2.5,  3.0,  4.0,  5.0,  6.0,  7.0,
         8.0,    9.0,   10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0,
         50.0,   60.0,  70.0, 80.0, 90.0]


    # Column 1 (converted by some factor)
    rm = [ 0.0,    0.011,  0.111,  1.112,  2.224,  3.336,  4.448,  6.672,
           8.896,  11.12,  17.79,  22.24,  27.80,  33.36,  44.48,  55.60, 
           66.72,  88.96,  111.2,  133.4,  177.9,  222.4,  278.0,  333.6,
           444.8,  556.0,  667.2,  778.4,  889.6,  1001.0, 1112.0, 1334.0,
           1779.0, 2224.0, 2780.0, 3336.0, 4448.0, 5560.0, 6672.0,
           7784.0, 8896.0, 10008.0] .* 1e3
    # converted to meters
    # GE /(10^12 rm) is vertical displacement in meters (applied load is 1kg)

    # Column 2
    GE = [ -33.6488, -33.64, -33.56, -32.75, -31.86, -30.98, -30.12, -28.44, -26.87, -25.41,
           -21.80, -20.02, -18.36, -17.18, -15.71, -14.91, -14.41, -13.69, -13.01,
           -12.31, -10.95, -9.757, -8.519, -7.533, -6.131, -5.237, -4.660, -4.272,
           -3.999, -3.798, -3.640, -3.392, -2.999, -2.619, -2.103, -1.530, -0.292,
            0.848,  1.676,  2.083,  2.057,  1.643];
    return T.(rm), T.(GE)
end

"""

    build_greenintegrand(distance::Vector{T}, 
        greenintegrand_coeffs::Vector{T}) where {T<:AbstractFloat}

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function build_greenintegrand(
    distance::Vector{T},
    greenintegrand_coeffs::Vector{T},
) where {T<:AbstractFloat}

    greenintegrand_interp = linear_interpolation(distance, greenintegrand_coeffs)
    compute_greenintegrand_entry_r(r::T) = get_loadgreen(
        r, distance, greenintegrand_coeffs, greenintegrand_interp)
    greenintegrand_function(x::T, y::T) = compute_greenintegrand_entry_r( get_r(x, y) )
    return greenintegrand_function
end

"""

    get_loadgreen(r::T, rm::Vector{T}, greenintegrand_coeffs::Vector{T},     
        interp_greenintegrand_::Interpolations.Extrapolation) where {T<:AbstractFloat}

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function get_loadgreen(
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

    dx, dy = Omega.dx, Omega.dy
    elasticgreen = similar(Omega.X)

    @inbounds for i = 1:Omega.Nx, j = 1:Omega.Ny
        p = i - Omega.Mx - 1
        q = j - Omega.My - 1
        elasticgreen[i, j] = quadrature2D(
            greenintegrand_function,
            quad_support,
            quad_coeffs,
            p*dx,
            (p+1)*dx,
            q*dy,
            (q+1)*dy,
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
    @inbounds for i=1:n
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
    @inbounds for i=1:n, j=1:n
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

"""

    kernelpromote(X, arraykernel)

Promote X to the kernel (`Array` or `CuArray`) specified by `arraykernel`.
"""
function kernelpromote(X::M, arraykernel) where {M<:AbstractArray{T}} where {T<:Real}
    if isa(X, arraykernel)
        return X
    else
        return arraykernel(X)
    end
end

function kernelpromote(X::Vector{M}, arraykernel) where {M<:AbstractArray{T}} where {T<:Real}
    if isa(X[1], arraykernel)
        return X
    else
        return [arraykernel(x) for x in X]
    end
end

function convert2CuArray(X::Vector)
    return [CuArray(x) for x in X]
end

function convert2Array(X::Vector)
    return [Array(x) for x in X]
end

function copystructs2cpu(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
) where {T<:AbstractFloat}

    Omega_cpu = ComputationDomain(Omega.Wx, Omega.Wy, Omega.Nx, Omega.Ny, use_cuda = false)

    p_cpu = MultilayerEarth(
        Omega_cpu,
        c;
        layer_boundaries = Array(p.layer_boundaries),
        layers_density = Array(p.layers_density),
        layer_viscosities = Array(p.layer_viscosities),
    )

    return Omega_cpu, p_cpu
end

function samesize_conv(X::XMatrix, Y::XMatrix, Omega::ComputationDomain)
    if iseven(Omega.Ny)
        i1 = Omega.My
    else
        i1 = Omega.My+1
    end
    i2 = 2*Omega.Ny-1-Omega.My

    if iseven(Omega.Nx)
        j1 = Omega.Mx
    else
        j1 = Omega.Mx+1
    end
    j2 = 2*Omega.Nx-1-Omega.Mx

    return view( conv(X, Y), i1:i2, j1:j2 )
end