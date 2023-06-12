XMatrix = Union{Matrix{T}, CuArray{T, 2}} where {T<:Real}

#########################################################
# Computation domain
#########################################################
"""
    ComputationDomain

Return a struct containing all information related to geometry of the domain
and potentially used parallelism. To initialize one with `2*W` and `2^n` grid cells:

```julia
Omega = ComputationDomain(W, n)
```
"""
struct ComputationDomain{T<:AbstractFloat}
    Wx::T                       # Domain length in x (m)
    Wy::T                       # Domain length in y (m)
    N::Int                      # Average number of grid points along one dimension
    N2::Int                     # N/2
    dx::T                       # Spatial discretization in x
    dy::T                       # Spatial discretization in y
    x::Vector{T}
    y::Vector{T}
    X::XMatrix
    Y::XMatrix
    R::XMatrix
    Theta::XMatrix
    Lat::XMatrix
    Lon::XMatrix
    K::XMatrix
    projection_correction::Bool
    null::XMatrix         # a zero matrix of size Nx x Ny
    pseudodiff::XMatrix   # pseudodiff operator
    harmonic::XMatrix     # harmonic operator
    biharmonic::XMatrix   # biharmonic operator
    use_cuda::Bool
    arraykernel                     # Array or CuArray depending on chosen hardware
end

function ComputationDomain(
    W::T,
    n::Int;
    use_cuda::Bool = false,
    lat0::T = T(-71.0),
    lon0::T = T(0.0),
    projection_correction::Bool = true,
) where {T<:AbstractFloat}

    # Geometry
    Wx, Wy = W, W
    N = 2^n
    N2 = Int(floor(N/2))
    dx = T(2*Wx) / N
    dy = T(2*Wy) / N
    x = collect(-Wx+dx:dx:Wx)
    y = collect(-Wy+dy:dy:Wy)
    X, Y = meshgrid(x, y)
    R = get_r.(X, Y)
    Theta = dist2angulardist.(R)
    Lat, Lon = stereo2latlon(X, Y, lat0, lon0)
    
    arraykernel = use_cuda ? CuArray : Array
    K = kernelpromote(scalefactor(deg2rad.(Lat), deg2rad.(Lon),
        deg2rad(lat0), deg2rad(lon0)), arraykernel)
    null = fill(T(0.0), N, N)
    
    # Differential operators in Fourier space
    pseudodiff, harmonic, biharmonic = get_differential_fourier(W, N2)
    pseudodiff[1, 1] = mean([pseudodiff[1,2], pseudodiff[2,1]])
    pseudodiff, harmonic, biharmonic = kernelpromote(
            [pseudodiff, harmonic, biharmonic], arraykernel)
    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.

    return ComputationDomain(
        Wx, Wy, N, N2,
        dx, dy, x, y,
        X, Y, R, Theta,
        Lat, Lon, K, projection_correction, null,
        pseudodiff, harmonic, biharmonic,
        use_cuda, arraykernel,
    )
end

#########################################################
# Physical constants
#########################################################
"""
    PhysicalConstants

Return a struct containing important physical constants.
"""
Base.@kwdef struct PhysicalConstants{T<:AbstractFloat}
    mE::T = 5.972e24                           # Earth's mass (kg)
    r_equator::T = 6.371e6                     # Earth radius at equator (m)
    r_pole::T = 6.357e6                        # Earth radius at pole (m)
    A_ocean::T = 3.625e14                      # Ocean surface (m) as in Goelzer (2020) before Eq. (9)
    g::T = 9.81                                # Mean Earth acceleration at surface (m/s^2)
    G::T = 6.674e-11                           # Gravity constant (m^3 kg^-1 s^-2)
    seconds_per_year::T = 60^2 * 24 * 365.25   # (s)
    rho_ice::T = 0.910e3                       # (kg/m^3)
    rho_water::T = 1.023e3                     # (kg/m^3)
    rho_seawater::T = 1.023e3                  # (kg/m^3)
    rho_core::T = 13.1e3                       # Density of Earth's core (kg m^-3)
    rho_topastheno::T = 3.3e3                  # Mean density of solid-Earth surface (kg m^-3)
    rho_litho::T = 3.0e3                       # Mean density of solid-Earth surface (kg m^-3)
end
# Note: rho_0 and rho_1 are chosen such that g(pole) ≈ 9.81

#########################################################
# Multi-layer Earth
#########################################################
"""
    MultilayerEarth

Return a struct containing all information related to the radially layered structure of the solid Earth and
its parameters.
"""
mutable struct MultilayerEarth{T<:AbstractFloat}
    mean_gravity::T
    mean_density::T
    effective_viscosity::XMatrix
    litho_thickness::XMatrix
    litho_rigidity::XMatrix
    litho_poissonratio::T
    layers_density::Vector{T}
    layer_viscosities::Array{T, 3}
    layer_boundaries::Array{T, 3}
end


litho_rigidity = 5e24               # (N*m)
litho_youngmodulus = 6.6e10         # (N/m^2)
litho_poissonratio = 0.5            # (1)
layers_density = [3.3e3]            # (kg/m^3)
layer_viscosities = [1e19, 1e21]     # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
layer_boundaries = [88e3, 400e3]
# 88 km: beginning of asthenosphere (Bueler 2007).
# 400 km: beginning of homogenous half-space (Ivins 2022, Fig 12).

# litho_rigidity = 5e24u"N*m"
# litho_youngmodulus = 6.6e10u"N / m^2"
# litho_poissonratio = 0.5
# layers_density = [3.3e3]u"kg / m^3"
# layer_viscosities = [1e19, 1e21]u"Pa*s"      # (Bueler 2007, Ivins 2022, Fig 12 WAIS)
# layer_boundaries = [88e3, 400e3]u"m"

function MultilayerEarth(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T};
    layer_boundaries::A = layer_boundaries,
    layers_density::Vector{T} = layers_density,
    layer_viscosities::B = layer_viscosities,
    litho_youngmodulus::C = litho_youngmodulus,
    litho_poissonratio::D = litho_poissonratio,
) where {
    T<:AbstractFloat,
    A<:Union{Vector{T}, Array{T, 3}},
    B<:Union{Vector{T}, Array{T, 3}},
    C<:Union{T, XMatrix},
    D<:Union{T, XMatrix},
}

    if layer_boundaries isa Vector{<:Real}
        layer_boundaries = matrify_vectorconstant(layer_boundaries, Omega.N)
    end
    if layer_viscosities isa Vector{<:Real}
        layer_viscosities = matrify_vectorconstant(layer_viscosities, Omega.N)
    end

    litho_thickness = layer_boundaries[:, :, 1]
    litho_rigidity = get_rigidity.(
        litho_thickness,
        litho_youngmodulus,
        litho_poissonratio,
    )

    layers_thickness = diff( layer_boundaries, dims=3 )
    # pseudodiff = kernelpromote(Omega.pseudodiff, Omega.arraykernel)
    effective_viscosity = get_effective_viscosity(
        Omega,
        layer_viscosities,
        layers_thickness,
    )

    mean_density = fill(mean(layers_density), Omega.N, Omega.N)

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
        layer_viscosities,
        layer_boundaries,
    )

end

"""
    RefGeoState

Return a struct containing the reference geostate. We define the geostate to be all quantities related to sea-level.
"""
struct RefGeoState{T<:AbstractFloat}
    H_ice::XMatrix          # reference height of ice column
    H_water::XMatrix        # reference height of water column
    b::XMatrix              # reference bedrock position
    z0::XMatrix             # reference height to allow external sea-level forcing
    sealevel::XMatrix       # reference sealevel field
    sle_af::T               # reference sl-equivalent of ice volume above floatation
    V_pov::T                # reference potential ocean volume
    V_den::T                # reference potential ocean volume associated with V_den
    conservation_term::T    # a term for mass conservation
end

"""
    GeoState

Return a mutable struct containing the geostate which will be updated over the simulation.
"""
mutable struct GeoState{T<:AbstractFloat}
    H_ice::XMatrix          # current height of ice column
    H_water::XMatrix        # current height of water column
    b::XMatrix              # vertical bedrock position
    geoid::XMatrix          # current geoid displacement
    sealevel::XMatrix       # current sealevel field
    V_af::T                 # ice volume above floatation
    sle_af::T               # sl-equivalent of ice volume above floatation
    slc_af::T               # sl-contribution of Vice above floatation
    V_pov::T                # current potential ocean volume
    slc_pov::T              # sea-level contribution associated with V_pov
    V_den::T                # potential ocean volume associated with density differences
    slc_den::T              # sea-level contribution associated with V_den
    slc::T                  # total sealevel contribution
    countupdates::Int       # count the updates of the geostate
    dt::T                   # update step
end

"""

    PrecomputedFastiso(Omega::ComputationDomain, c::PhysicalConstants, p::MultilayerEarth)

Return a `struct` containing pre-computed tools to perform forward-stepping of the model, namely:
 - elasticgreen::XMatrix
 - fourier_elasticgreen::XMatrix{Complex{T}}
 - pfft::AbstractFFTs.Plan
 - pifft::AbstractFFTs.ScaledPlan
 - Dx::XMatrix
 - Dy::XMatrix
 - Dxx::XMatrix
 - Dyy::XMatrix
 - Dxy::XMatrix
 - negligible_gradD::Bool
 - rhog::T
 - geoidgreen::XMatrix
"""
struct PrecomputedFastiso{T<:AbstractFloat}
    elasticgreen::XMatrix
    fourier_elasticgreen::XMatrix{Complex{T}}
    pfft::AbstractFFTs.Plan
    pifft::AbstractFFTs.ScaledPlan
    Dx::XMatrix
    Dy::XMatrix
    Dxx::XMatrix
    Dyy::XMatrix
    Dxy::XMatrix
    negligible_gradD::Bool
    rhog::T
    geoidgreen::XMatrix
end


function PrecomputedFastiso(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T};
    quad_precision::Int = 4,
) where {T<:AbstractFloat}

    # Elastic response variables
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elasticgreen = get_elasticgreen(Omega, greenintegrand_function, quad_support, quad_coeffs)

    # Space-derivatives of rigidity
    D = kernelpromote(p.litho_rigidity, Array)
    Dx = mixed_fdx(D, Omega.dx)
    Dy = mixed_fdy(D, Omega.dy)
    Dxx = mixed_fdxx(D, Omega.dx)
    Dyy = mixed_fdyy(D, Omega.dy)
    Dxy = mixed_fdy( mixed_fdx(D, Omega.dx), Omega.dy )

    omega_zeros = fill(T(0.0), Omega.N, Omega.N)
    zero_tol = 1e-2
    negligible_gradD = isapprox(Dx, omega_zeros, atol = zero_tol) &
                        isapprox(Dy, omega_zeros, atol = zero_tol) &
                        isapprox(Dxx, omega_zeros, atol = zero_tol) &
                        isapprox(Dyy, omega_zeros, atol = zero_tol) &
                        isapprox(Dxy, omega_zeros, atol = zero_tol)
    
    # FFT plans depening on CPU vs. GPU usage
    if Omega.use_cuda
        Xgpu = CuArray(Omega.X)
        p1, p2 = CUDA.CUFFT.plan_fft(Xgpu), CUDA.CUFFT.plan_ifft(Xgpu)
        Dx, Dy, Dxx, Dyy, Dxy = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy])
    else
        p1, p2 = plan_fft(Omega.X), plan_ifft(Omega.X)
    end

    rhog = p.mean_density .* c.g
    geoidgreen = get_geoidgreen(Omega, c)

    return PrecomputedFastiso(
        elasticgreen, fft(elasticgreen),
        p1, p2,
        Dx, Dy, Dxx, Dyy, Dxy, negligible_gradD,
        rhog,
        kernelpromote(geoidgreen, Omega.arraykernel),
    )
end

"""
    SuperStruct()

Return a struct containing all the other structs needed for the forward integration of the model:
 - Omega::ComputationDomain{T}
 - c::PhysicalConstants{T}
 - p::MultilayerEarth{T}
 - tools::PrecomputedFastiso{T}
 - Hice::Interpolations.Extrapolation
 - Hice_cpu::Interpolations.Extrapolation
 - eta::Interpolations.Extrapolation
 - eta_cpu::Interpolations.Extrapolation
 - refgeostate::RefGeoState{T}
 - geostate::GeoState{T}
 - interactive_geostate::Bool
"""
struct SuperStruct{T<:AbstractFloat}
    Omega::ComputationDomain{T}
    c::PhysicalConstants{T}
    p::MultilayerEarth{T}
    tools::PrecomputedFastiso{T}
    Hice::Interpolations.Extrapolation
    Hice_cpu::Interpolations.Extrapolation
    eta::Interpolations.Extrapolation
    eta_cpu::Interpolations.Extrapolation
    refgeostate::RefGeoState{T}
    geostate::GeoState{T}
    interactive_geostate::Bool
end

function SuperStruct(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{Matrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{<:XMatrix},
    interactive_geostate::Bool;
    geoid_0::Matrix{T} = copy(Omega.null),
    sealevel_0::Matrix{T} = copy(Omega.null),
    H_ice_ref::Matrix{T} = copy(Omega.null),
    H_water_ref::Matrix{T} = copy(Omega.null),
    b_ref::Matrix{T} = copy(Omega.null),
) where {T<:AbstractFloat}

    geoid_0, sealevel_0, H_ice_ref, H_water_ref, b_ref = kernelpromote(
        [geoid_0, sealevel_0, H_ice_ref, H_water_ref, b_ref], Omega.arraykernel)

    tools = PrecomputedFastiso(Omega, c, p)
    Hice = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel) )
    Hice_cpu = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Array) )
    eta = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Omega.arraykernel) )
    eta_cpu = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Array) )

    refgeostate = RefGeoState(
        H_ice_ref, H_water_ref, b_ref,
        copy(sealevel_0),   # z0
        sealevel_0,         # sealevel
        T(0.0),             # sle_af
        T(0.0),             # V_pov
        T(0.0),             # V_den
        T(0.0),             # conservation_term
    )
    geostate = GeoState(
        Hice(0.0),                  # ice column
        copy(H_water_ref),          # water column
        copy(b_ref),                # bedrock position
        geoid_0,                    # geoid perturbation
        copy(sealevel_0),           # reference for external sl-forcing
        T(0.0), T(0.0), T(0.0),     # V_af terms
        T(0.0), T(0.0),             # V_pov terms
        T(0.0), T(0.0),             # V_den terms
        T(0.0),                     # total sl-contribution & conservation term
        0, years2seconds(10.0),     # countupdates, update step
    )
    return SuperStruct(Omega, c, p, tools, Hice, Hice_cpu, eta, eta_cpu,
        refgeostate, geostate, interactive_geostate)
end

"""

    FastisoResults(Omega::ComputationDomain, c::PhysicalConstants, p::MultilayerEarth)

Return a `struct` containing the results of forward integration:
 - `t_out` the time output vector
 - `u3D_elastic` the elastic response over `t_out`
 - `u3D_viscous` the viscous response over `t_out`
 - `dudt3D_viscous` the displacement rate over `t_out`
 - `geoid3D` the geoid response over `t_out`
 - `Hice` an interpolator of the ice thickness over time
 - `eta` an interpolator of the upper-mantle viscosity over time
"""
struct FastisoResults{T<:AbstractFloat}
    t_out::Vector{T}
    tools::PrecomputedFastiso{T}
    viscous::Vector{Matrix{T}}
    displacement_rate::Vector{Matrix{T}}
    elastic::Vector{Matrix{T}}
    geoid::Vector{Matrix{T}}
    sealevel::Vector{Matrix{T}}
    Hice::Interpolations.Extrapolation
    eta::Interpolations.Extrapolation
end

