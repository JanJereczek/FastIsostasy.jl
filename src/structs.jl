#########################################################
# Convenience
#########################################################
KernelMatrix{T} = Union{Matrix{T}, CuMatrix{T}} where {T<:AbstractFloat}
mutable struct PreAllocated{T<:AbstractFloat, M<:KernelMatrix{T}}
    rhs::M
    uxx::M
    uyy::M
    ux::M
    uxy::M
    Mxxxx::M
    Myyyy::M
    Mxyx::M
    Mxyxy::M
end

#########################################################
# Computation domain
#########################################################
"""
    ComputationDomain
    ComputationDomain(W, n)
    ComputationDomain(Wx, Wy, Nx, Ny)

Return a struct containing all information related to geometry of the domain
and potentially used parallelism. To initialize one with `2*W` and `2^n` grid cells:

```julia
Omega = ComputationDomain(W, n)
```

If a rectangular domain is needed, run:

```julia
Omega = ComputationDomain(Wx, Wy, Nx, Ny)
```
"""
struct ComputationDomain{T<:AbstractFloat, M<:KernelMatrix{T}}
    Wx::T                           # Domain length in x (m)
    Wy::T                           # Domain length in y (m)
    Nx::Int                         # Number of grid points in x-dimension
    Ny::Int                         # Number of grid points in y-dimension
    Mx::Int                         # Nx/2
    My::Int                         # Ny/2
    dx::T                           # Spatial discretization in x
    dy::T                           # Spatial discretization in y
    Dx::M
    Dy::M
    x::Vector{T}
    y::Vector{T}
    X::M
    Y::M
    i1::Int
    i2::Int
    j1::Int
    j2::Int
    R::M
    Theta::M
    Lat::M
    Lon::M
    K::M
    projection_correction::Bool
    null::M         # a zero matrix of size Nx x Ny
    pseudodiff::M   # pseudodiff operator
    use_cuda::Bool
    arraykernel::Any                # Array or CuArray depending on chosen hardware
    bc!::Function                   # Boundary conditions
end

function ComputationDomain(W::T, n::Int; kwargs...) where {T<:AbstractFloat}
    Wx, Wy = W, W
    Nx, Ny = 2^n, 2^n
    return ComputationDomain(Wx, Wy, Nx, Ny; kwargs...)
end

function ComputationDomain(
    Wx::T,
    Wy::T,
    Nx::Int,
    Ny::Int;
    bc!::Function = corner_bc!,
    use_cuda::Bool = false,
    lat0::T = T(-71.0),
    lon0::T = T(0.0),
    projection_correction::Bool = true,
) where {T<:AbstractFloat}

    # Geometry
    Mx, My = Nx ÷ 2, Ny ÷ 2
    dx = 2*Wx / Nx
    dy = 2*Wy / Ny
    x = collect(-Wx+dx:dx:Wx)
    y = collect(-Wy+dy:dy:Wy)
    X, Y = meshgrid(x, y)
    null = fill(T(0), Nx, Ny)
    R = get_r.(X, Y)
    Theta = dist2angulardist.(R)
    Lat, Lon = stereo2latlon(X, Y, lat0, lon0)

    arraykernel = use_cuda ? CuArray : Array
    if projection_correction
        K = scalefactor(deg2rad.(Lat), deg2rad.(Lon), deg2rad(lat0), deg2rad(lon0))
    else
        K = fill(T(1), Nx, Ny)
    end

    # Differential operators in Fourier space
    pseudodiff, harmonic = get_differential_fourier(Wx, Wy, Nx, Ny)
    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.
    pseudodiff[1, 1] = mean([pseudodiff[1,2], pseudodiff[2,1]])
    
    X, Y, null, R, Theta, Lat, Lon, K, pseudodiff = kernelpromote(
        [X, Y, null, R, Theta, Lat, Lon, K, pseudodiff], arraykernel)

    # Precompute indices for samesize_conv()
    if iseven(Nx)
        i1 = Mx
    else
        i1 = Mx+1
    end
    i2 = 2*Nx-1-Mx

    if iseven(Ny)
        j1 = My
    else
        j1 = My+1
    end
    j2 = 2*Ny-1-My

    return ComputationDomain(Wx, Wy, Nx, Ny, Mx, My, dx, dy, K .* dx, K .* dy,
        x, y, X, Y, i1, i2, j1, j2, R, Theta, Lat, Lon, K, projection_correction,
        null, pseudodiff, use_cuda, arraykernel, bc!)
end

#########################################################
# Physical constants
#########################################################
"""
    PhysicalConstants

Return a struct containing important physical constants.
Comes with default values that can however be changed by the user (e.g. ice density).
"""
Base.@kwdef struct PhysicalConstants{T<:AbstractFloat}
    mE::T = 5.972e24                        # Earth's mass (kg)
    r_equator::T = 6.371e6                  # Earth radius at equator (m)
    r_pole::T = 6.357e6                     # Earth radius at pole (m)
    A_ocean::T = 3.625e14                   # Ocean surface (m) as in Goelzer (2020) before Eq. (9)
    g::T = 9.8                              # Mean Earth acceleration at surface (m/s^2)
    G::T = 6.674e-11                        # Gravity constant (m^3 kg^-1 s^-2)
    seconds_per_year::T = SECONDS_PER_YEAR  # (s)
    rho_ice::T = 0.910e3                    # (kg/m^3)
    rho_water::T = 1e3                      # (kg/m^3)
    rho_seawater::T = 1.023e3               # (kg/m^3)
    rho_uppermantle::T = 3.7e3              # Mean density of topmost upper mantle (kg m^-3)
    rho_litho::T = 2.6e3                    # Mean density of lithosphere (kg m^-3)
end

#########################################################
# Multi-layer Earth
#########################################################
"""
    ReferenceEarthModel
Return a struct with vectors containing the:
 - radius (distance from Earth center),
 - depth (distance from Earth surface),
 - density,
 - P-wave velocities,
 - S-wave velocities,
which are typically used to characterize the properties of a spherically symmetrical solid Earth.
"""
struct ReferenceEarthModel{T<:AbstractFloat}
    radius::Vector{T}
    depth::Vector{T}
    density::Vector{T}
    Vpv::Vector{T}
    Vph::Vector{T}
    Vsv::Vector{T}
    Vsh::Vector{T}
end

"""
    LateralVariability

Return a struct containing all information related to the radially layered structure of the solid Earth and
its parameters.
"""
mutable struct LateralVariability{T<:AbstractFloat, M<:KernelMatrix{T}}
    effective_viscosity::M
    litho_thickness::M
    litho_rigidity::M
    litho_poissonratio::T
    mantle_poissonratio::T
    layer_viscosities::Array{T, 3}
    layer_boundaries::Array{T, 3}
end

litho_rigidity = 5e24               # (N*m)
litho_youngmodulus = 6.6e10         # (N/m^2)
litho_poissonratio = 0.28           # (1)
mantle_poissonratio = 0.28          # (1)
layer_densities = [3.3e3]           # (kg/m^3)
layer_viscosities = [1e19, 1e21]    # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
layer_boundaries = [88e3, 400e3]
# 88 km: beginning of asthenosphere (Bueler 2007).
# 400 km: beginning of homogenous half-space (Ivins 2022, Fig 12).

# litho_rigidity = 5e24u"N*m"
# litho_youngmodulus = 6.6e10u"N / m^2"
# litho_poissonratio = 0.5
# layer_densities = [3.3e3]u"kg / m^3"
# layer_viscosities = [1e19, 1e21]u"Pa*s"      # (Bueler 2007, Ivins 2022, Fig 12 WAIS)
# layer_boundaries = [88e3, 400e3]u"m"

function LateralVariability(
    Omega::ComputationDomain{T, M};
    layer_boundaries::A = layer_boundaries,
    layer_viscosities::B = layer_viscosities,
    litho_youngmodulus::T = litho_youngmodulus,
    litho_poissonratio::T = litho_poissonratio,
    mantle_poissonratio::T = mantle_poissonratio,
) where {
    T<:AbstractFloat, M<:KernelMatrix{T},
    A<:Union{Vector{T}, Array{T, 3}},
    B<:Union{Vector{T}, Array{T, 3}},
}

    if layer_boundaries isa Vector{<:Real}
        layer_boundaries = matrify(layer_boundaries, Omega.Nx, Omega.Ny)
    end
    if layer_viscosities isa Vector{<:Real}
        layer_viscosities = matrify(layer_viscosities, Omega.Nx, Omega.Ny)
    end

    litho_thickness = layer_boundaries[:, :, 1]
    litho_rigidity = get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)

    layers_thickness = diff(layer_boundaries, dims=3)
    effective_viscosity = get_effective_viscosity(
        Omega, layer_viscosities, layers_thickness, mantle_poissonratio)

    litho_thickness, litho_rigidity, effective_viscosity = kernelpromote(
        [litho_thickness, litho_rigidity, effective_viscosity], Omega.arraykernel)
    return LateralVariability(
        effective_viscosity,
        litho_thickness, litho_rigidity, litho_poissonratio,
        mantle_poissonratio, layer_viscosities, layer_boundaries,
    )

end

#########################################################
# Geostate
#########################################################
"""
    RefGeoState

Return a struct containing the reference [`GeoState`](@ref).
"""
struct RefGeoState{T<:AbstractFloat, M<:KernelMatrix{T}}
    u::M                    # viscous displacement
    ue::M                   # elastic displacement
    H_ice::M                # reference height of ice column
    H_water::M              # reference height of water column
    b::M                    # reference bedrock position
    z0::M                   # reference height to allow external sea-level forcing
    sealevel::M             # reference sealevel field
    sle_af::T               # reference sl-equivalent of ice volume above floatation
    V_pov::T                # reference potential ocean volume
    V_den::T                # reference potential ocean volume associated with V_den
    conservation_term::T    # a term for mass conservation
end

"""
    GeoState

Return a mutable struct containing the geostate which will be updated over the simulation.
The geostate contains all the states of the [`FastIsoProblem`] to be solved.
"""
mutable struct GeoState{T<:AbstractFloat, M<:KernelMatrix{T}}
    u::M                    # viscous displacement
    dudt::M                 # viscous displacement rate
    ue::M                   # elastic displacement
    H_ice::M                # current height of ice column
    H_water::M              # current height of water column
    b::M                    # vertical bedrock position
    geoid::M                # current geoid displacement
    sealevel::M             # current sealevel field
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

#########################################################
# FastIsostasy
#########################################################
"""
    FastIsoTools(Omega::ComputationDomain, c::PhysicalConstants, p::LateralVariability)

Return a `struct` containing pre-computed tools to perform forward-stepping of the model, namely:
 - elasticgreen::KernelMatrix{T}
 - fourier_elasticgreen::KernelMatrix{T}{Complex{T}}
 - pfft::AbstractFFTs.Plan
 - pifft::AbstractFFTs.ScaledPlan
 - geoidgreen::KernelMatrix{T}
"""
struct FastIsoTools{T<:AbstractFloat, M<:KernelMatrix{T},
    P1<:AbstractFFTs.Plan{Complex{T}}, P2<:AbstractFFTs.Plan{Complex{T}}}
    elasticgreen::M
    geoidgreen::M
    pfft::P1
    pifft::AbstractFFTs.ScaledPlan{Complex{T}, P2, T}
    Hice::Interpolations.Extrapolation{M, 1, Interpolations.GriddedInterpolation{M, 1, Vector{M},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, Flat{Nothing}}
    eta::Interpolations.Extrapolation{M, 1, Interpolations.GriddedInterpolation{M, 1, Vector{M},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, Flat{Nothing}}
    prealloc::PreAllocated{T, M}
end

function FastIsoTools(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{<:KernelMatrix{T}};
    quad_precision::Int = 4,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # Elastic response variables
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elasticgreen = get_elasticgreen(Omega, greenintegrand_function, quad_support, quad_coeffs)
    geoidgreen = get_geoidgreen(Omega, c)

    # FFT plans depening on CPU vs. GPU usage
    if Omega.use_cuda
        Xgpu = CuArray(Omega.X)
        p1, p2 = CUFFT.plan_fft(Xgpu), CUFFT.plan_ifft(Xgpu)
        # Dx, Dy, Dxx, Dyy, Dxy = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy])
    else
        p1, p2 = plan_fft(Omega.X), plan_ifft(Omega.X)
    end

    # rhog = p.uppermantle_density .* c.g
    Hice = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel); extrapolation_bc=Flat())
    eta = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Omega.arraykernel); extrapolation_bc=Flat())

    prealloc = PreAllocated([null(Omega) for i in eachindex(fieldnames(PreAllocated))]...)
    return FastIsoTools(Omega.arraykernel(elasticgreen), Omega.arraykernel(geoidgreen),
        p1, p2, Hice, eta, prealloc)
end

null(Omega::ComputationDomain) = copy(Omega.arraykernel(Omega.null))

"""
    FastIsoOutputs

Return a struct containing the fields that were saved over a [`FastIsoProblem`].
"""
mutable struct FastIsoOutputs{T<:AbstractFloat}
    t::Vector{T}
    u::Vector{Matrix{T}}
    dudt::Vector{Matrix{T}}
    ue::Vector{Matrix{T}}
    geoid::Vector{Matrix{T}}
    sealevel::Vector{Matrix{T}}
    computation_time::Float64
end

struct SimpleEuler end

"""
    FastIsoProblem()

Return a struct containing all the other structs needed for the forward integration of the model:
 - Omega::ComputationDomain{T, M}
 - c::PhysicalConstants{T}
 - p::LateralVariability{T, M}
 - tools::FastIsoTools{T, M}
 - Hice::Interpolations.Extrapolation
 - eta::Interpolations.Extrapolation
 - refgeostate::RefGeoState{T, M}
 - geostate::GeoState{T, M}
 - interactive_sealevel::Bool
"""
struct FastIsoProblem{T<:AbstractFloat, M<:KernelMatrix{T}}
    Omega::ComputationDomain{T, M}
    c::PhysicalConstants{T}
    p::LateralVariability{T, M}
    tools::FastIsoTools{T, M}
    refgeostate::RefGeoState{T, M}
    geostate::GeoState{T, M}
    interactive_sealevel::Bool
    internal_loadupdate::Bool
    diffeq::NamedTuple
    verbose::Bool
    out::FastIsoOutputs
end

function FastIsoProblem(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool;
    kwargs...,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    # Creating some placeholders in case of an external update of the load.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [null(Omega), null(Omega)]
    return FastIsoProblem(Omega, c, p, t_out, interactive_sealevel,
        t_Hice_snapshots, Hice_snapshots, internal_loadupdate = false; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool,
    Hice::KernelMatrix{T};
    kwargs...,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    # Constant interpolator in case viscosity is fixed over time.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [Hice, Hice]
    return FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, 
        t_Hice_snapshots, Hice_snapshots, internal_loadupdate = true; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool,
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}};
    kwargs...,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    # Constant interpolator in case viscosity is fixed over time.
    t_eta_snapshots = [extrema(t_out)...]
    eta_snapshots = [p.effective_viscosity, p.effective_viscosity]
    return FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, 
        t_Hice_snapshots, Hice_snapshots, t_eta_snapshots, eta_snapshots,
        internal_loadupdate = true; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool,
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{<:KernelMatrix{T}};
    diffeq::NamedTuple = (alg = BS3(), reltol = 1e-3),
    verbose::Bool = false,
    internal_loadupdate::Bool = true,
    u_0::M = null(Omega),
    dudt_0::M = null(Omega),
    ue_0::M = null(Omega),
    geoid_0::M = null(Omega),
    z_0::M = null(Omega),
    sealevel_0::M = null(Omega),
    H_ice_0::M = null(Omega),
    H_water_0::M = null(Omega),
    b_0::M = null(Omega),
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    if !isa(diffeq.alg, OrdinaryDiffEqAlgorithm) && !isa(diffeq.alg, SimpleEuler)
        error("Provided algorithm for solving ODE is not supported.")
    end

    tools = FastIsoTools(Omega, c, p, t_Hice_snapshots, Hice_snapshots,
        t_eta_snapshots, eta_snapshots)

    refgeostate = RefGeoState(
        u_0, ue_0,          # viscous and elastic displacement
        H_ice_0, H_water_0, b_0,    # ice & liquid water column
        z_0,                        # z0 (useful for external sealevel forcing)
        sealevel_0,                 # sealevel
        T(0.0),                     # sle_af
        T(0.0),                     # V_pov
        T(0.0),                     # V_den
        T(0.0),                     # conservation_term
    )
    
    geostate = GeoState(
        copy(u_0), copy(dudt_0),    # viscous displacement and associated rate
        copy(ue_0),                 # elastic displacement
        tools.Hice(0.0),            # ice column
        copy(H_water_0),            # water column
        copy(b_0),                  # bedrock position
        geoid_0,                    # geoid perturbation
        copy(sealevel_0),           # reference for external sl-forcing
        T(0.0), T(0.0), T(0.0),     # V_af terms
        T(0.0), T(0.0),             # V_pov terms
        T(0.0), T(0.0),             # V_den terms
        T(0.0),                     # total sl-contribution & conservation term
        0, years2seconds(10.0),     # countupdates, update step
    )

    # Extend the vector with a zero at beginning if not already the case
    # Typically needed for post-processing.
    if !(0 in t_out)
        pushfirst!(0, t_out)
    end

    out = init_results(Omega, t_out)
    
    return FastIsoProblem(Omega, c, p, tools, refgeostate, geostate, interactive_sealevel,
        internal_loadupdate, diffeq, verbose, out)
end