"""

    meshgrid(x, y)

Return a 2D meshgrid spanned by `x, y`.
"""
@inline function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return one_y * x', (one_x * y')'
end

@inline function convert2CuArray(X::Vector)
    return [CuArray(x) for x in X]
end

@inline function convert2Array(X::Vector)
    return [Array(x) for x in X]
end

"""

    init_domain(L, n)

Initialize a square computational domain with length `2*L` and `2^n+1` grid cells.
"""
@inline function init_domain(
    L::T,
    n::Int;
    use_cuda=false::Bool
) where {T<:AbstractFloat}

    N = 2^n
    N2 = Int(floor(N/2))
    h = T(2*L) / N
    x = collect(-L+h:h:L)
    X, Y = meshgrid(x, x)
    distance, loadresponse_coeffs = get_loadresponse_coeffs(T)
    loadresponse_matrix, loadresponse_function = build_loadresponse_matrix(X, Y, distance, loadresponse_coeffs)
    pseudodiff, harmonic, biharmonic = get_differential_fourier(L, N2)

    if use_cuda
        pseudodiff, harmonic, biharmonic = convert2CuArray([pseudodiff, harmonic, biharmonic])
    end
    
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
        pseudodiff,
        harmonic,
        biharmonic,
        use_cuda,
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
    harmonic_coeffs::AbstractMatrix{T}
    biharmonic_coeffs::AbstractMatrix{T}
    use_cuda::Bool
end

"""
    get_differential_fourier(L, N2)

Compute the matrices representing the differential operators in the fourier space.
"""
@inline function get_differential_fourier(
    L::T,
    N2::Int,
) where {T<:Real}
    mu = T(π / L)
    raw_coeffs = mu .* T.( vcat(0:N2, N2-1:-1:1) )
    x_coeffs, y_coeffs = raw_coeffs, raw_coeffs
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    harmonic_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(harmonic_coeffs)
    biharmonic_coeffs = harmonic_coeffs .^ 2
    return pseudodiff_coeffs, harmonic_coeffs, biharmonic_coeffs
end
#####################################################
############### Physical constants ##################
#####################################################

g = 9.81                                # Mean Earth acceleration at surface (m/s^2)
seconds_per_year = 60^2 * 24 * 365.25   # (s)
ice_density = 0.910e3                   # (kg/m^3)
r_equator = 6.371e6                     # Earth radius at equator (m)
r_pole = 6.357e6                        # Earth radius at pole (m)
G = 6.674e-11                           # Gravity constant (m^3 kg^-1 s^-2)
mE = 5.972e24                           # Earth's mass (kg)
rho_0 = 13.1e3                          # Density of Earth's core (kg m^-3)
rho_1 = 3.0e3                           # Mean density of solid-Earth surface (kg m^-3)
# Note: rho_0 and rho_1 are chosen such that g(pole) ≈ 9.81

"""
    init_physical_constants()

Return struct containing physical constants.
"""
@inline function init_physical_constants(;T::Type=Float64, ice_density = ice_density)
    return PhysicalConstants(
        T(g),
        T(seconds_per_year),
        T(ice_density),
        T(r_equator),
        T(r_pole),
        T(G),
        T(mE),
        T(rho_0),
        T(rho_1),
    )
end

struct PhysicalConstants{T<:AbstractFloat}
    g::T
    seconds_per_year::T
    ice_density::T
    r_equator::T
    r_pole::T
    G::T
    mE::T
    rho_0::T
    rho_1::T
end

function years2seconds(t::T) where {T<:AbstractFloat}
    return t * seconds_per_year
end

function seconds2years(t::T) where {T<:AbstractFloat}
    return t / seconds_per_year
end

#####################################################
############# Solid Earth parameters ################
#####################################################

lithosphere_rigidity = 5e24     # N*m
layers_density = [3.3e3]        # kg/m^3
layers_viscosity = [1e19, 1e21] # Pa*s (Ivins 2022, Fig 12 WAIS)
layers_begin = [88e3, 400e3]
# 88 km: beginning of asthenosphere (Bueler 2007).
# 400 km: beginning of homogenous half-space (Ivins 2022, Fig 12).

"""

    init_multilayer_earth(
        Omega::ComputationDomain{T};
        lithosphere_rigidity<:Union{Vector{T}, Vector{AbstractMatrix{T}}},
        layers_density::Vector{T},
        layers_viscosity<:Union{Vector{T}, Vector{AbstractMatrix{T}}},
        layers_begin::Vector{T},
    ) where {T<:AbstractFloat}

Return struct with solid-Earth parameters for mutliple channel layers and a halfspace.
"""
@inline function init_multilayer_earth(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T};
    lithosphere_rigidity::ScalarOrMatrix = lithosphere_rigidity,
    layers_density::Vector{T} = layers_density,
    layers_viscosity::VectorOr3DArray = layers_viscosity,
    layers_begin::Vector{T} = layers_begin,
) where {
    T<:AbstractFloat,
    ScalarOrMatrix<:Union{T, AbstractMatrix{T}},
    VectorOr3DArray<:Union{Vector{T}, AbstractArray{T, 3}},
}

    if lithosphere_rigidity isa Real
        lithosphere_rigidity = matrify_constant(lithosphere_rigidity, Omega.N)
    end
    if layers_viscosity isa Vector
        layers_viscosity = matrify_vectorconstant(layers_viscosity, Omega.N)
    end

    # Earth acceleration over depth z.
    # Use the pole radius because GIA most important at poles.
    gr(r) = 4*π/3*c.G*c.rho_0* r - π*c.G*(c.rho_0 - c.rho_1) * r^2/c.r_pole
    gz(z) = gr(c.r_pole - z)

    layers_thickness = diff( layers_begin )
    layers_mean_gravity = 0.5 .*(gz.(layers_begin[1:end-1]) + gz.(layers_begin[2:end]))
    mean_gravity = (layers_thickness ./ (sum(layers_thickness)))' * layers_mean_gravity
    mean_density = (layers_thickness ./ (sum(layers_thickness)))' * layers_density

    if Omega.use_cuda
        pseudodiff_coeffs = Array(Omega.pseudodiff_coeffs)
    else
        pseudodiff_coeffs = Omega.pseudodiff_coeffs
    end
    effective_viscosity = get_effective_viscosity(
        Omega,
        layers_viscosity,
        layers_thickness,
        pseudodiff_coeffs,
    )
    if Omega.use_cuda
        lithosphere_rigidity, effective_viscosity = convert2CuArray(
            [lithosphere_rigidity, effective_viscosity]
        )
    end

    return MultilayerEarth(
        mean_gravity,
        mean_density,
        effective_viscosity,
        lithosphere_rigidity,
        layers_density,
        layers_viscosity,
        layers_begin,
    )

end

struct MultilayerEarth{T<:AbstractFloat}
    mean_gravity::T
    mean_density::T
    effective_viscosity::AbstractMatrix{T}
    lithosphere_rigidity::AbstractMatrix{T}
    layers_density::Vector{T}
    layers_viscosity::AbstractArray{T, 3}
    layers_begin::Vector{T}
end

"""

    matrify_vectorconstant(x, N)

Generate a vector of constant matrices from a vector of constants.
"""
@inline function matrify_vectorconstant(x::Vector{T}, N::Int) where {T<:AbstractFloat}
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
@inline function matrify_constant(x::T, N::Int) where {T<:AbstractFloat}
    return fill(x, N, N)
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
@inline function get_effective_viscosity(
    Omega::ComputationDomain{T},
    layers_viscosity::AbstractArray{T, 3},
    layers_thickness::Vector{T},
    pseudodiff_coeffs::AbstractMatrix{T},
) where {T<:AbstractFloat}

    effective_viscosity = layers_viscosity[:, :, end]    # begin with half space
    p1, p2 = plan_fft(effective_viscosity), plan_ifft(effective_viscosity)
    for i in axes(layers_viscosity, 3)[1:end-1]
        channel_viscosity = layers_viscosity[:, :, end - i]
        channel_thickness = layers_thickness[end - i + 1]
        viscosity_ratio = get_viscosity_ratio(channel_viscosity, effective_viscosity)
        viscosity_scaling = three_layer_scaling(
            Omega,
            pseudodiff_coeffs,
            viscosity_ratio,
            channel_thickness,
        )
        copy!( 
            effective_viscosity,
            real.( p2 * (( p1 * effective_viscosity ) .* viscosity_scaling)),
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
@inline function get_viscosity_ratio(
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
@inline function three_layer_scaling(
    Omega::ComputationDomain{T},
    kappa::Matrix{T},
    visc_ratio::Matrix{T},
    channel_thickness::T,
) where {T<:AbstractFloat}

    visc_scaling = zeros(T, size(kappa)...)
    for i in axes(kappa, 1), j in axes(kappa, 2)

        k = π / Omega.L  # kappa[i, j]                 # (1/m)
        vr = visc_ratio[i, j]
        C, S = hyperbolic_channel_coeffs(channel_thickness, k)
        
        num1 = 2 * vr * C * S
        num2 = (1 - vr ^ 2) * channel_thickness^2 * k ^ 2
        num3 = vr ^ 2 * S ^ 2 + C ^ 2

        denum1 = (vr + 1/vr) * C * S
        denum2 = (vr - 1/vr) * channel_thickness * k
        denum3 = S^2 + C^2
        
        visc_scaling[i, j] = (num1 + num2 + num3) / (denum1 + denum2 + denum3)
    end
    return visc_scaling
end

# @inline function three_layer_scaling(
#     kappa::AbstractMatrix{T},
#     visc_ratio::AbstractMatrix{T},
#     channel_thickness::T,
# ) where {T<:AbstractFloat}

#     C(k) = cosh(channel_thickness * k)
#     S(k) = sinh(channel_thickness * k)
#     C, S = C.(kappa), S.(kappa)

#     num1 = 2 .* visc_ratio .* C .* S
#     num2 = (1 .- visc_ratio .^ 2) .* channel_thickness .^ 2 .* kappa .^ 2
#     num3 = visc_ratio .^ 2 .* S .^ 2 + C .^ 2

#     denum1 = (visc_ratio + 1 ./ visc_ratio) .* C .* S
#     denum2 = (visc_ratio - 1 ./ visc_ratio) .* channel_thickness .* kappa
#     denum3 = S .^ 2 + C .^ 2

#     return (num1 + num2 + num3) ./ (denum1 + denum2 + denum3)
# end

"""

    hyperbolic_channel_coeffs(channel_thickness, kappa)

Return hyperbolic coefficients for equivalent viscosity computation.
"""
@inline function hyperbolic_channel_coeffs(
    channel_thickness::T,
    kappa::T,
) where {T<:AbstractFloat}
    return cosh(channel_thickness * kappa), sinh(channel_thickness * kappa)
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
############## Copy main structs CPU ################
#####################################################

function copystructs2cpu(
    Omega::ComputationDomain,
    p::MultilayerEarth,
    c::PhysicalConstants,
)

    T = typeof( Omega.L )
    n = Int( round( log2(Omega.N) ) )
    Omega_cpu = init_domain(Omega.L, n, use_cuda = false)

    p_cpu = init_multilayer_earth(
        Omega_cpu,
        c;
        lithosphere_rigidity = Array(p.lithosphere_rigidity),
        layers_density = p.layers_density,
        layers_viscosity = p.layers_viscosity,
        layers_begin = p.layers_begin,
    )
    return Omega_cpu, p_cpu
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