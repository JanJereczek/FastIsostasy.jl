#####################################################
# Array utils
#####################################################

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

#####################################################
# Domain utils
#####################################################

"""

    get_r(x, y)

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:Real} = LinearAlgebra.norm([x, y])

"""

    meshgrid(x, y)

Return a 2D meshgrid spanned by `x, y`.
"""
@inline function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return one_y * x', (one_x * y')'
end

"""

    init_domain(L, n)

Initialize a square computational domain with length `2*L` and `2^n` grid cells.
"""
@inline function init_domain(
    L::T,
    n::Int;
    use_cuda=false::Bool
) where {T<:AbstractFloat}

    Lx, Ly = L, L
    N = 2^n
    N2 = Int(floor(N/2))
    dx = T(2*Lx) / N
    dy = T(2*Ly) / N
    x = collect(-Lx+dx:dx:Lx)
    y = collect(-Ly+dy:dy:Ly)
    X, Y = meshgrid(x, y)
    distance, loadresponse_coeffs = get_loadresponse_coeffs(T)
    loadresponse_matrix, loadresponse_function = build_loadresponse_matrix(
        X, Y,
        distance,
        loadresponse_coeffs,
    )
    pseudodiff, harmonic, biharmonic = get_differential_fourier(L, N2)

    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.
    pseudodiff[1, 1] = mean([pseudodiff[1,2], pseudodiff[2,1]])

    if use_cuda
        pseudodiff, harmonic, biharmonic = convert2CuArray(
            [pseudodiff, harmonic, biharmonic])
    end
    
    return ComputationDomain(
        Lx,
        Ly,
        N,
        N2,
        dx,
        dy,
        x,
        y,
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
    Lx::T
    Ly::T
    N::Int
    N2::Int
    dx::T
    dy::T
    x::Vector{T}
    y::Vector{T}
    X::AbstractMatrix{T}
    Y::AbstractMatrix{T}
    loadresponse_matrix::AbstractMatrix{T}
    loadresponse_function::Function
    pseudodiff_coeffs::AbstractMatrix{T}
    harmonic_coeffs::AbstractMatrix{T}
    biharmonic_coeffs::AbstractMatrix{T}
    use_cuda::Bool
end

#####################################################
# Differential utils
#####################################################

# Fourier
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

@inline function precomp_fourier_dxdy(
    M::AbstractMatrix{T},
    L1::T,
    L2::T,
) where {T<:AbstractFloat}
    n1, n2 = size(M)
    k1 = 2 * π / L1 .* vcat(0:n1/2-1, 0, -n1/2+1:-1)
    k2 = 2 * π / L2 .* vcat(0:n2/2-1, 0, -n2/2+1:-1)
    p1, p2 = plan_fft(k1), plan_fft(k2)
    ip1, ip2 = plan_ifft(k1), plan_ifft(k2)
    return k1, k2, p1, p2, ip1, ip2
end

@inline function fourier_dnx(
    M::AbstractMatrix{T},
    k1::Vector{T},
    p1::AbstractFFTs.Plan,
    ip1::AbstractFFTs.ScaledPlan,
) where {T<:AbstractFloat}
    return vcat( [real.(ip1 * ( ( im .* k1 ) .^ n .* (p1 * M[i, :]) ))' for i in axes(M,1)]... )
end

# FDM in x, 1st order
@inline function central_fdx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, 3:n2) - view(M, :, 1:n2-2)) ./ (2*h)
end

@inline function forward_fdx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return (view(M, :, 2) - view(M, :, 1)) ./ h
end

@inline function backward_fdx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, n2) - view(M, :, n2-1)) ./ h
end

# FDM in y, 1st order
@inline function mixed_fdx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return cat( forward_fdx(M,h), central_fdx(M,h), backward_fdx(M,h), dims=2 )
end

@inline function central_fdy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, 3:n1, :) - view(M, 1:n1-2, :)) ./ (2*h)
end

@inline function forward_fdy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return (view(M, 2, :) - view(M, 1, :)) ./ h
end

@inline function backward_fdy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, n1, :) - view(M, n1-1, :)) ./ h
end

@inline function mixed_fdy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return cat( forward_fdy(M,h)', central_fdy(M,h), backward_fdy(M,h)', dims=1 )
end

# FDM in x, 2nd order
@inline function central_fdxx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, 3:n2) - 2 .* view(M, :, 2:n2-1) + view(M, :, 1:n2-2)) ./ h^2
end

@inline function forward_fdxx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return (view(M, :, 3) - 2 .* view(M, :, 2) + view(M, :, 1)) ./ h^2
end

@inline function backward_fdxx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n2 = size(M, 2)
    return (view(M, :, n2) - 2 .* view(M, :, n2-1) + view(M, :, n2-2)) ./ h^2
end

@inline function mixed_fdxx(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return cat( forward_fdxx(M,h), central_fdxx(M,h), backward_fdxx(M,h), dims=2 )
end

# FDM in y, 2nd order
@inline function central_fdyy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, 3:n1, :) - 2 .* view(M, 2:n1-1, :) + view(M, 1:n1-2, :)) ./ h^2
end

@inline function forward_fdyy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return (view(M, 3, :) - 2 .* view(M, 2, :) + view(M, 1, :)) ./ h^2
end

@inline function backward_fdyy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    n1 = size(M, 1)
    return (view(M, n1, :) - 2 .* view(M, n1-1, :) + view(M, n1-2, :)) ./ h^2
end

@inline function mixed_fdyy(M::AbstractMatrix{T}, h::T) where {T<:AbstractFloat}
    return cat( forward_fdyy(M,h)', central_fdyy(M,h), backward_fdyy(M,h)', dims=1 )
end

@inline function gauss_distr(x::T, mu::Vector{T}, sigma::Matrix{T}) where {T<:AbstractFloat}
    k = length(mu)
    return (2 * π)^(k/2) * det(sigma) * exp( -0.5 * (x .- mu)' * inv(sigma) * (x .- mu) )
end

@inline function gauss_distr(
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
# Physical constants
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

function m_per_sec2mm_per_yr(dudt::T) where {T<:AbstractFloat}
    return dudt * 1e3 * seconds_per_year
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
@inline function init_multilayer_earth(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T};
    layers_begin::A = layers_begin,
    layers_density::Vector{T} = layers_density,
    layers_viscosity::B = layers_viscosity,
    litho_youngmodulus::C = litho_youngmodulus,
    litho_poissonratio::D = litho_poissonratio,
) where {
    T<:AbstractFloat,
    A<:Union{Vector{T}, AbstractArray{T, 3}},
    B<:Union{Vector{T}, AbstractArray{T, 3}},
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
    if Omega.use_cuda
        pseudodiff_coeffs = Array(Omega.pseudodiff_coeffs)
    else
        pseudodiff_coeffs = Omega.pseudodiff_coeffs
    end
    effective_viscosity = get_effective_viscosity(
        Omega,
        layers_viscosity,
        layers_thickness,
        # pseudodiff_coeffs,
    )

    # mean_density = get_matrix_mean_density(layers_thickness, layers_density)
    mean_density = fill(layers_density[1], Omega.N, Omega.N)

    if Omega.use_cuda
        litho_rigidity, effective_viscosity, mean_density = convert2CuArray(
            [litho_rigidity, effective_viscosity, mean_density]
        )
    end

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

struct MultilayerEarth{T<:AbstractFloat}
    mean_gravity::T
    mean_density::T
    effective_viscosity::AbstractMatrix{T}
    litho_thickness::AbstractMatrix{T}
    litho_rigidity::AbstractMatrix{T}
    litho_poissonratio::T
    layers_density::Vector{T}
    layers_viscosity::AbstractArray{T, 3}
    layers_begin::AbstractArray{T, 3}
end

@inline function get_rigidity(
    t::T,
    E::T,
    nu::T,
) where {T<:AbstractFloat}
    return (E * t^3) / (12 * (1 - nu^2))
end

@inline function get_matrix_mean_density(
    layers_thickness::AbstractArray{T, 3},
    layers_density::Vector{T},
) where {T<:AbstractFloat}
    mean_density = zeros(T, size(layers_thickness)[1:2])
    for i in axes(layers_thickness, 1), j in axes(layers_thickness, 2)
        mean_density[i, j] = get_mean_density(layers_thickness[i, j, :], layers_density)
    end
    return mean_density
end

@inline function get_mean_density(
    layers_thickness::Vector{T},
    layers_density::Vector{T},
) where {T<:AbstractFloat}
    return sum( (layers_thickness ./ (sum(layers_thickness)))' * layers_density )
end

@inline function matrified_mean_gravity()
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

@inline function get_mean_gravity(
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
@inline function get_effective_viscosity(
    Omega::ComputationDomain{T},
    layers_viscosity::AbstractArray{T, 3},
    layers_thickness::AbstractArray{T, 3},
    # pseudodiff_coeffs::AbstractMatrix{T},
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
            # pseudodiff_coeffs,
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

    loginterp_viscosity(tvec, layers_viscosity, layers_thickness, pseudodiff_coeffs)

Compute a log-interpolator of the equivalent viscosity from provided viscosity
fields `layers_viscosity` at time stamps `tvec`.
"""
@inline function loginterp_viscosity(
    tvec::AbstractVector{T},
    layers_viscosity::AbstractArray{T, 4},
    layers_thickness::AbstractArray{T, 3},
    pseudodiff_coeffs::AbstractMatrix,
) where {T<:AbstractFloat}
    n1, n2, n3, nt = size(layers_viscosity)
    log_eqviscosity = [fill(T(0.0), n1, n2) for k in 1:nt]

    [log_eqviscosity[k] .= log10.(get_effective_viscosity(
        layers_viscosity[:, :, :, k],
        layers_thickness,
        pseudodiff_coeffs,
    )) for k in 1:nt]

    log_interp = linear_interpolation(tvec, log_eqviscosity)
    visc_interp(t) = 10 .^ log_interp(t)
    return visc_interp
end

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
# Load response utils
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

    h = Omega.dx
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
        )
    end
    return integrated_loadresponse
end

#####################################################
# Quadrature utils
#####################################################

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


#####################################################
# Kernel utils
#####################################################

@inline function convert2CuArray(X::Vector)
    return [CuArray(x) for x in X]
end

@inline function convert2Array(X::Vector)
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