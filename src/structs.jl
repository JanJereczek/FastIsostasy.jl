#########################################################
# Convenience
#########################################################
"""
    KernelMatrix

An allias for `Union{Matrix{T}, CuMatrix{T}} where {T<:AbstractFloat}`.
"""
KernelMatrix{T} = Union{Matrix{T}, CuMatrix{T}} where {T<:AbstractFloat}
ComplexMatrix{T} = Union{Matrix{C}, CuMatrix{C}} where {T<:AbstractFloat, C<:Complex{T}}
ForwardPlan{T} = Union{
    cFFTWPlan{Complex{T}, -1, true, 2, Tuple{Int64, Int64}}, 
    CUFFT.cCuFFTPlan{Complex{T}, -1, true, 2}
} where {T<:AbstractFloat}
InversePlan{T} = Union{
    AbstractFFTs.ScaledPlan{Complex{T}, cFFTWPlan{Complex{T}, 1, true, 2, UnitRange{Int64}}, T},
    AbstractFFTs.ScaledPlan{Complex{T}, CUFFT.cCuFFTPlan{Complex{T}, 1, true, 2}, T}
} where {T<:AbstractFloat}
mutable struct PreAllocated{T<:AbstractFloat, M<:KernelMatrix{T}, C<:ComplexMatrix{T}}
    rhs::M
    uxx::M
    uyy::M
    ux::M
    uxy::M
    Mxx::M
    Myy::M
    Mxy::M
    Mxxxx::M
    Myyyy::M
    Mxyx::M
    Mxyxy::M
    fftrhs::C
    ifftrhs::C
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
struct ComputationDomain{T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    Wx::T                       # Domain half-width in x (m)
    Wy::T                       # Domain half-width in y (m)
    Nx::Int                     # Number of grid points in x-dimension
    Ny::Int                     # Number of grid points in y-dimension
    Mx::Int                     # Nx/2
    My::Int                     # Ny/2
    dx::T                       # Spatial discretization in x
    dy::T                       # Spatial discretization in y
    x::Vector{T}                # spanning vector in x-dimension
    y::Vector{T}                # spanning vector in y-dimension
    X::L
    Y::L
    i1::Int                     # indices for samesize_conv
    i2::Int
    j1::Int
    j2::Int
    R::L                        # euclidean distance from center
    Theta::L                    # colatitude
    Lat::L
    Lon::L
    K::M                        # Length distortion matrix
    Dx::M                       # dx matrix accounting for distortion
    Dy::M                       # dy matrix accounting for distortion
    A::M                        # surface matrix accounting for distortion
    correct_distortion::Bool
    null::M                     # a zero matrix of size Nx x Ny
    pseudodiff::M               # pseudodiff operator as matrix (Hadamard product)
    biharmonic::M               # biharmonic operator as matrix (Hadamard product)
    use_cuda::Bool
    arraykernel::Any            # Array or CuArray depending on chosen hardware
    bc!::Function               # Boundary conditions
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
    lat0::T = T(-90.0),
    lon0::T = T(0.0),
    correct_distortion::Bool = true,
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
    Lat, Lon = stereo2latlon(X, Y, lat0, lon0)

    arraykernel = use_cuda ? CuArray : Array
    if correct_distortion
        K = scalefactor(deg2rad.(Lat), deg2rad.(Lon), deg2rad(lat0), deg2rad(lon0))
    else
        K = fill(T(1), Nx, Ny)
    end
    Theta = dist2angulardist.(K .* R)

    # Differential operators in Fourier space
    pseudodiff, _, biharmonic = get_differential_fourier(Wx, Wy, Nx, Ny)

    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.
    pseudodiff[1, 1] = 1e-3 * mean([pseudodiff[1,2], pseudodiff[2,1]])
    
    null, K, pseudodiff, biharmonic = kernelpromote([null, K, pseudodiff, biharmonic], arraykernel)

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

    return ComputationDomain(Wx, Wy, Nx, Ny, Mx, My, dx, dy, x, y, X, Y, i1, i2, j1, j2,
        R, Theta, Lat, Lon, K, K .* dx, K .* dy, (dx * dy) .* K .^ 2, correct_distortion,
        null, pseudodiff, biharmonic, use_cuda, arraykernel, bc!)
end

#########################################################
# Physical constants
#########################################################
"""
    PhysicalConstants

Return a struct containing important physical constants.
Comes with default values that can however be changed by the user, for instance by running:

```julia
c = PhysicalConstants(rho_ice = 0.93)   # (kg/m^3)
```

All constants are given in SI units (kilogram, meter, second).
"""
Base.@kwdef struct PhysicalConstants{T<:AbstractFloat}
    mE::T = 5.972e24                        # Earth's mass (kg)
    r_equator::T = 6371e3                   # Earth radius at equator (m)
    r_pole::T = 6357e3                      # Earth radius at pole (m)
    A_ocean_pd::T = 3.625e14                # Ocean surface (m) as in Goelzer (2020) before Eq. (9)
    g::T = 9.8                              # Mean Earth acceleration at surface (m/s^2)
    G::T = 6.674e-11                        # Gravity constant (m^3 kg^-1 s^-2)
    seconds_per_year::T = SECONDS_PER_YEAR  # (s)
    rho_ice::T = 0.910e3                    # (kg/m^3)
    rho_water::T = 1e3                      # (kg/m^3)
    rho_seawater::T = 1.023e3               # (kg/m^3)
    # rho_uppermantle::T = 3.7e3            # Mean density of topmost upper mantle (kg m^-3)
    # rho_litho::T = 2.6e3                  # Mean density of lithosphere (kg m^-3)
    rho_uppermantle::T = 3.4e3              # Mean density of topmost upper mantle (kg m^-3)
    rho_litho::T = 3.2e3                    # Mean density of lithosphere (kg m^-3)
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
    LayeredEarth(Omega; layer_boundaries, layer_viscosities)

Return a struct containing all information related to the lateral variability of
solid-Earth parameters. To initialize with values other than default, run:

```julia
Omega = ComputationDomain(3000e3, 7)
lb = [100e3, 300e3]
lv = [1e19, 1e21]
p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
```

which initializes a lithosphere of thickness \$T_1 = 100 \\mathrm{km}\$, a viscous
channel between \$T_1\$ and \$T_2 = 200 \\mathrm{km}\$ and a viscous halfspace starting
at \$T_2\$. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `Omega::ComputationDomain`.
"""
mutable struct LayeredEarth{T<:AbstractFloat, M<:KernelMatrix{T}}
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

function LayeredEarth(
    Omega::ComputationDomain{T, L, M};
    layer_boundaries::A = layer_boundaries,
    layer_viscosities::B = layer_viscosities,
    litho_youngmodulus::T = litho_youngmodulus,
    litho_poissonratio::T = litho_poissonratio,
    mantle_poissonratio::T = mantle_poissonratio,
) where {
    T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T},
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
    return LayeredEarth(
        effective_viscosity,
        litho_thickness, litho_rigidity, litho_poissonratio,
        mantle_poissonratio, layer_viscosities, layer_boundaries,
    )

end

#########################################################
# Geostate
#########################################################
abstract type GeoState end

"""
    ReferenceState

Return a struct containing the reference state.
"""
struct ReferenceState{T<:AbstractFloat, M<:KernelMatrix{T}} <: GeoState
    u::M                    # viscous displacement
    ue::M                   # elastic displacement
    H_ice::M                # ref height of ice column
    H_water::M              # ref height of water column
    b::M                    # ref bedrock position
    bsl::T                  # ref barystatic sea level
    seasurfaceheight::M     # ref seasurfaceheight field
    V_af::T                 # ref sl-equivalent of ice volume above floatation
    V_pov::T                # ref potential ocean volume
    V_den::T                # ref potential ocean volume associated with V_den
    maskgrounded::KernelMatrix{<:Bool} # mask for grounded ice
end

mutable struct ColumnAnomalies{T<:AbstractFloat, M<:KernelMatrix{T}}
    load::M
    litho::M
    mantle::M
    full::M
end

function ColumnAnomalies(Omega)
    zero_columnanoms = [null(Omega) for _ in eachindex(fieldnames(ColumnAnomalies))]
    return ColumnAnomalies(zero_columnanoms...)
end

"""
    CurrentState

Return a mutable struct containing the geostate which will be updated over the simulation.
The geostate contains all the states of the [`FastIsoProblem`] to be solved.
"""
mutable struct CurrentState{T<:AbstractFloat, M<:KernelMatrix{T}} <: GeoState
    u::M                    # viscous displacement
    dudt::M                 # viscous displacement rate
    ue::M                   # elastic displacement
    H_ice::M                # current height of ice column
    H_water::M              # current height of water column
    columnanoms::ColumnAnomalies{T, M}
    b::M                    # vertical bedrock position
    bsl::T                  # barystatic sea level
    geoid::M                # current geoid displacement
    seasurfaceheight::M     # current seasurfaceheight field
    V_af::T                 # V contribution from ice above floatation
    V_pov::T                # V contribution from bedrock adjustment
    V_den::T                # V contribution from diff between melt- and saltwater density
    maskgrounded::KernelMatrix{<:Bool} # mask for grounded ice
    osc::OceanSurfaceChange
    countupdates::Int       # count the updates of the geostate
    Δt::T                   # update step
end

# Initialise CurrentState from ReferenceState
function CurrentState(Omega::ComputationDomain{T, L, M}, ref::ReferenceState{T, M};
    Δt = years2seconds(10.0)) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix}
    return CurrentState(
        copy(ref.u), null(Omega), copy(ref.ue),     # u, dudt, ue
        copy(ref.H_ice), copy(ref.H_water),         # H_ice, H_water
        ColumnAnomalies(Omega), copy(ref.b),        # columnanoms, b
        copy(ref.bsl), null(Omega), copy(ref.seasurfaceheight),  # b, bsl, seasurfaceheight
        copy(ref.V_af), copy(ref.V_pov), copy(ref.V_den),       # V_af, V_pov, V_den
        copy(ref.maskgrounded), OceanSurfaceChange(z0 = ref.bsl),
        0, Δt,
    )
end

#########################################################
# FastIsostasy
#########################################################
"""
    FastIsoTools(Omega, c, p)
Return a `struct` containing pre-computed tools to perform forward-stepping of the model.
This includes the Green's functions for the computation of the lithosphere and geoid
displacement, plans for FFTs, interpolators of the load and the viscosity over time and
preallocated arrays.
"""
struct FastIsoTools{T<:AbstractFloat, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    elasticgreen::M
    geoidgreen::M
    pfft!::FP
    pifft!::IP
    Hice::Interpolations.Extrapolation{M, 1, Interpolations.GriddedInterpolation{M, 1, Vector{M},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, Flat{Nothing}}
    eta::Interpolations.Extrapolation{M, 1, Interpolations.GriddedInterpolation{M, 1, Vector{M},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, Flat{Nothing}}
    prealloc::PreAllocated{T, M, C}
end

# CUDA.CUFFT.cCuFFTPlan{ComplexF64, -1, true, 2}
# AbstractFFTs.ScaledPlan{ComplexF64, CUDA.CUFFT.cCuFFTPlan{ComplexF64, 1, true, 2}, Float64}
# FFTW.cFFTWPlan{ComplexF64, -1, true, 2, Tuple{Int64, Int64}}
# AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 2, UnitRange{Int64}}, Float64}

function FastIsoTools(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{<:KernelMatrix{T}};
    quad_precision::Int = 4,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}

    # Elastic response variables
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elasticgreen = get_elasticgreen(Omega, greenintegrand_function, quad_support, quad_coeffs)
    geoidgreen = get_geoidgreen(Omega, c)

    # FFT plans depening on CPU vs. GPU usage
    if Omega.use_cuda
        pfft! = CUFFT.plan_fft!(complex.(Omega.K))
        pifft! = CUFFT.plan_ifft!(complex.(Omega.K))
    else
        pfft! = plan_fft!(complex.(Omega.K))
        pifft! = plan_ifft!(complex.(Omega.K))
    end

    # rhog = p.uppermantle_density .* c.g
    Hice = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel); extrapolation_bc=Flat())
    eta = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Omega.arraykernel); extrapolation_bc=Flat())

    realmatrices = [null(Omega) for _ in eachindex(fieldnames(PreAllocated))[1:end-2]]
    cplxmatrices = [complex.(null(Omega)) for _ in 1:2]
    prealloc = PreAllocated(realmatrices..., cplxmatrices...)
    
    return FastIsoTools(Omega.arraykernel(elasticgreen), Omega.arraykernel(geoidgreen),
        pfft!, pifft!, Hice, eta, prealloc)
end

null(Omega::ComputationDomain) = copy(Omega.arraykernel(Omega.null))

"""
    FastIsoOutputs()

Return a struct containing the fields of viscous displacement, viscous displacement rate,
elastic displacement, geoid displacement, sea level and the computation time resulting
from solving a [`FastIsoProblem`](@ref).
"""
mutable struct FastIsoOutputs{T<:AbstractFloat, M<:Matrix{T}}
    t::Vector{T}
    u::Vector{M}
    dudt::Vector{M}
    ue::Vector{M}
    geoid::Vector{M}
    seasurfaceheight::Vector{M}
    Hice::Vector{M}
    computation_time::Float64
end

struct SimpleEuler end

"""
    FastIsoProblem(Omega, c, p, t_out, interactive_sealevel)
    FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hice)
    FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, t_Hice, Hice)

Return a struct containing all the other structs needed for the forward integration of the
model over `Omega::ComputationDomain` with parameters `c::PhysicalConstants` and
`p::LayeredEarth`. The outputs are stored at `t_out::Vector{<:AbstractFloat}`.
"""
struct FastIsoProblem{T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    Omega::ComputationDomain{T, L, M}
    c::PhysicalConstants{T}
    p::LayeredEarth{T, M}
    tools::FastIsoTools{T, M, C, FP, IP}
    refgeostate::ReferenceState{T, M}
    geostate::CurrentState{T, M}
    interactive_sealevel::Bool
    internal_loadupdate::Bool
    neglect_litho_gradients::Bool
    diffeq::NamedTuple
    verbose::Bool
    out::FastIsoOutputs{T, L}
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool;
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Creating some placeholders in case of an external update of the load.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [null(Omega), null(Omega)]
    return FastIsoProblem(Omega, c, p, t_out, interactive_sealevel,
        t_Hice_snapshots, Hice_snapshots, internal_loadupdate = false; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool,
    Hice::KernelMatrix{T};
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Constant interpolator in case viscosity is fixed over time.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [Hice, Hice]
    return FastIsoProblem(Omega, c, p, t_out, interactive_sealevel,
        t_Hice_snapshots, Hice_snapshots, internal_loadupdate = true; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool,
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}};
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Constant interpolator in case viscosity is fixed over time.
    t_eta_snapshots = [extrema(t_out)...]
    eta_snapshots = [p.effective_viscosity, p.effective_viscosity]
    return FastIsoProblem(Omega, c, p, t_out, interactive_sealevel,
        t_Hice_snapshots, Hice_snapshots, t_eta_snapshots, eta_snapshots,
        internal_loadupdate = true; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    interactive_sealevel::Bool,
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{<:KernelMatrix{T}};
    diffeq::NamedTuple = (alg = Tsit5(), reltol = 1e-3),
    verbose::Bool = false,
    internal_loadupdate::Bool = true,
    neglect_litho_gradients::Bool = false,
    u_0::KernelMatrix{T} = null(Omega),
    ue_0::KernelMatrix{T} = null(Omega),
    seasurfaceheight_0::KernelMatrix{T} = null(Omega),
    b_0::KernelMatrix{T} = null(Omega),
    bsl_0::T = 0.0,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}

    if !isa(diffeq.alg, OrdinaryDiffEqAlgorithm) && !isa(diffeq.alg, SimpleEuler)
        error("Provided algorithm for solving ODE is not supported.")
    end

    tools = FastIsoTools(Omega, c, t_Hice_snapshots, Hice_snapshots,
        t_eta_snapshots, eta_snapshots)

    # Initialise the reference state
    H_ice_0 = Hice_snapshots[1]
    u_0, ue_0, seasurfaceheight_0, b_0, H_ice_0 = kernelpromote([u_0, ue_0,
        seasurfaceheight_0, b_0, H_ice_0], Omega.arraykernel)
    maskgrounded = height_above_floatation(H_ice_0, b_0, seasurfaceheight_0,
        c.rho_seawater, c.rho_ice) .> 0
    if !Omega.use_cuda
        maskgrounded = collect(maskgrounded)    # use Matrix{Bool} rather than BitMatrix
    end
    H_water_0 = watercolumn(maskgrounded, b_0, seasurfaceheight_0)

    refgeostate = ReferenceState(
        u_0, ue_0,
        H_ice_0, H_water_0,
        b_0, bsl_0, seasurfaceheight_0,
        T(0.0), T(0.0), T(0.0),     # V_af, V_pov, V_den
        maskgrounded,
    )
    geostate = CurrentState(Omega, refgeostate)
    out = init_results(Omega, t_out)
    
    return FastIsoProblem(Omega, c, p, tools, refgeostate, geostate, interactive_sealevel,
        internal_loadupdate, neglect_litho_gradients, diffeq, verbose, out)
end


function Base.show(io::IO, ::MIME"text/plain", fip::FastIsoProblem)
    Omega, p = fip.Omega, fip.p
    println(io, "FastIsoProblem")
    descriptors = [
        "Wx, Wy" => [Omega.Wx, Omega.Wy],
        "dx, dy" => [Omega.dx, Omega.dy],
        "extrema(effective viscosity)" => extrema(p.effective_viscosity),
        "extrema(lithospheric thickness)" => extrema(p.litho_thickness),
    ]
    padlen = maximum(length(d[1]) for d in descriptors) + 2
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end

function remake!(fip::FastIsoProblem)
    fip.geostate.u = null(fip.Omega) #fip.refgeostate.u
    fip.geostate.dudt = null(fip.Omega)
    fip.geostate.ue = null(fip.Omega) #fip.refgeostate.ue
    fip.geostate.geoid = null(fip.Omega)
    fip.geostate.seasurfaceheight = null(fip.Omega) #fip.refgeostate.seasurfaceheight
    fip.geostate.H_water = null(fip.Omega) #fip.refgeostate.H_water
    fip.geostate.H_ice = fip.tools.Hice(0.0)
    fip.geostate.b = fip.refgeostate.b
    fip.geostate.countupdates = 0
    fip.geostate.columnanoms = ColumnAnomalies(fip.Omega)

    out = init_results(fip.Omega, fip.out.t)
    fip.out.u = out.u
    fip.out.ue = out.ue
    return nothing
end