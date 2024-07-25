#########################################################
# Prealloc
#########################################################
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
    convo_out::M
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
    convo_offset::Int
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
    use_cuda::Bool
    arraykernel::Any            # Array or CuArray depending on chosen hardware
    bc_matrix::M
    nbc::T
    extended_bc_matrix::M
    extended_nbc::T
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
    use_cuda::Bool = false,
    lat0::T = T(-90.0),
    lon0::T = T(0.0),
    correct_distortion::Bool = true,
) where {T<:AbstractFloat}

    # Geometry
    Mx, My = Nx ÷ 2, Ny ÷ 2
    dx = 2*Wx / Nx
    dy = 2*Wy / Ny
    # x = collect(-Wx+dx:dx:Wx)
    # y = collect(-Wy+dy:dy:Wy)
    x = collect(range(-Wx+dx, stop = Wx, length = Nx))
    y = collect(range(-Wy+dy, stop = Wy, length = Ny))
    X, Y = meshgrid(x, y)
    null = fill(T(0), Nx, Ny)
    R = get_r.(X, Y)
    Lat, Lon = stereo2latlon(X, Y, lat0, lon0)

    if correct_distortion
        K = scalefactor(deg2rad.(Lat), deg2rad.(Lon), deg2rad(lat0), deg2rad(lon0))
    else
        K = fill(T(1), Nx, Ny)
    end
    Theta = dist2angulardist.(K .* R)

    # Differential operators in Fourier space
    pseudodiff, _, _ = get_differential_fourier(Wx, Wy, Nx, Ny)

    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.
    pseudodiff[1, 1] = 1e-3 * mean([pseudodiff[1,2], pseudodiff[2,1]])
    
    arraykernel = use_cuda ? CuArray : Array
    null, K, pseudodiff = kernelpromote([null, K, pseudodiff], arraykernel)

    i1, i2 = samesize_conv_indices(Nx, Mx)
    j1, j2 = samesize_conv_indices(Ny, My)
    convo_offset = (Ny - Nx) ÷ 2

    bc_matrix = arraykernel(corner_matrix(T, Nx, Ny))
    nbc = sum(bc_matrix)
    extended_bc_matrix = arraykernel(corner_matrix(T, 2*Nx-1, 2*Ny-1))
    extended_nbc = sum(bc_matrix)

    return ComputationDomain(Wx, Wy, Nx, Ny, Mx, My, dx, dy, x, y, X, Y, i1, i2, j1, j2, convo_offset,
        R, Theta, Lat, Lon, K, K .* dx, K .* dy, (dx * dy) .* K .^ 2, correct_distortion,
        null, pseudodiff, use_cuda, arraykernel, bc_matrix, nbc, extended_bc_matrix,
        extended_nbc)
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
    type = Float64
    mE::T = type(5.972e24)                        # Earth's mass (kg)
    r_equator::T = type(6371e3)                   # Earth radius at equator (m)
    r_pole::T = type(6357e3)                      # Earth radius at pole (m)
    A_ocean_pd::T = type(3.625e14)                # Ocean surface (m^2) as in Goelzer (2020) before Eq. (9)
    g::T = type(9.8)                              # Mean Earth acceleration at surface (m/s^2)
    G::T = type(6.674e-11)                        # Gravity constant (m^3 kg^-1 s^-2)
    seconds_per_year::T = type(SECONDS_PER_YEAR)  # (s)
    rho_ice::T = type(0.910e3)                    # (kg/m^3)
    rho_water::T = type(1e3)                      # (kg/m^3)
    rho_seawater::T = type(1.023e3)               # (kg/m^3)
    # rho_uppermantle::T = 3.7e3            # Mean density of topmost upper mantle (kg m^-3)
    # rho_litho::T = 2.6e3                  # Mean density of lithosphere (kg m^-3)
    rho_uppermantle::T = type(3.4e3)              # Mean density of topmost upper mantle (kg m^-3)
    rho_litho::T = type(3.2e3)                    # Mean density of lithosphere (kg m^-3)
end

#########################################################
# Earth model
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

which initializes a lithosphere of thickness ``T_1 = 100 \\mathrm{km}``, a viscous
channel between ``T_1``and ``T_2 = 200 \\mathrm{km}``and a viscous halfspace starting
at ``T_2``. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `Omega::ComputationDomain`.
"""
mutable struct LayeredEarth{T<:AbstractFloat, M<:KernelMatrix{T}}
    effective_viscosity::M
    litho_thickness::Matrix{T}
    litho_rigidity::M
    litho_poissonratio::T
    mantle_poissonratio::T
    layer_viscosities::Array{T, 3}
    layer_boundaries::Array{T, 3}
    tau::M
    litho_youngmodulus::T
    litho_shearmodulus::T
    rho_uppermantle::T
    rho_litho::T
end

function LayeredEarth(
    Omega::ComputationDomain{T, L, M};
    litho_thickness = nothing,
    layer_boundaries::A = T.([88e3, 400e3]),    # 88 km: asthenosphere, 400 km: half-space (Bueler 2007).
    layer_viscosities::B = T.([1e19, 1e21]),    # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
    litho_youngmodulus::T = T(6.6e10),          # (N/m^2)
    litho_poissonratio::T = T(0.28),
    mantle_poissonratio::T = T(0.28),
    tau = years2seconds(855.0),
    rho_uppermantle = T(3.4e3),                 # Mean density of topmost upper mantle (kg m^-3)
    rho_litho = T(3.2e3),                       # Mean density of lithosphere (kg m^-3)
    layering = "folded",
) where {
    T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T},
    A<:Union{Vector{T}, Array{T, 3}},
    B<:Union{Vector{T}, Array{T, 3}},
}

    if tau isa Real
        tau = fill(tau, Omega.Nx, Omega.Ny)
    end
    tau = kernelpromote(tau, Omega.arraykernel)

    if layer_boundaries isa Vector{<:Real}
        layer_boundaries = matrify(layer_boundaries, Omega.Nx, Omega.Ny)
    end
    if layer_viscosities isa Vector{<:Real}
        layer_viscosities = matrify(layer_viscosities, Omega.Nx, Omega.Ny)
    end

    if isnothing(litho_thickness)
        litho_thickness = layer_boundaries[:, :, 1]
    end
    litho_rigidity = get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)

    if layering == "equalizing"
        layers_thickness = diff(layer_boundaries, dims=3)
        effective_viscosity = get_effective_viscosity(
            Omega, layer_viscosities, layers_thickness, mantle_poissonratio)
    elseif layering == "embedded"
        layers_thickness = diff(layer_boundaries, dims=3)
        effective_viscosity = new_effective_viscosity(Omega, litho_thickness,
            layer_boundaries, layer_viscosities, layers_thickness, mantle_poissonratio)
    elseif layering == "folded"
        effective_viscosity, folded_layers, folded_viscosities =
            interpolated_effective_viscosity(Omega, layer_boundaries, layer_viscosities,
                litho_thickness, mantle_poissonratio)
    else
        throw(ArgumentError("Unknown layering type: $layering"))
    end

    litho_rigidity, effective_viscosity = kernelpromote(
        [litho_rigidity, effective_viscosity], Omega.arraykernel)
    
    litho_shearmodulus = litho_youngmodulus / (2 * (1 + litho_poissonratio))

    return LayeredEarth(
        effective_viscosity,
        litho_thickness, litho_rigidity, litho_poissonratio,
        mantle_poissonratio, folded_viscosities, folded_layers,
        tau, litho_youngmodulus, litho_shearmodulus, rho_uppermantle, rho_litho,
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
    z_ss::M     # ref z_ss field
    V_af::T                 # ref sl-equivalent of ice volume above floatation
    V_pov::T                # ref potential ocean volume
    V_den::T                # ref potential ocean volume associated with V_den
    maskgrounded::KernelMatrix{<:Bool}  # mask for grounded ice
    maskocean::KernelMatrix{<:Bool}     # mask for ocean
    maskactive::KernelMatrix{<:Bool}
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
    u_eq::M                 # equilibrium dispalcement
    ucorner::T              # corner displacement of the domain
    H_ice::M                # current height of ice column
    H_water::M              # current height of water column
    columnanoms::ColumnAnomalies{T, M}
    b::M                    # vertical bedrock position
    bsl::T                  # barystatic sea level
    dz_ss::M                # current z_ss perturbation
    z_ss::M                 # current z_ss field
    V_af::T                 # V contribution from ice above floatation
    V_pov::T                # V contribution from bedrock adjustment
    V_den::T                # V contribution from diff between melt- and saltwater density
    maskgrounded::KernelMatrix{<:Bool}  # mask for grounded ice
    maskocean::KernelMatrix{<:Bool}     # mask for ocean
    osc::OceanSurfaceChange{T}
    countupdates::Int       # count the updates of the geostate
    k::Int                  # index of the t_out segment
end

# Initialise CurrentState from ReferenceState
function CurrentState(Omega::ComputationDomain{T, L, M}, ref::ReferenceState{T, M}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    return CurrentState(
        copy(ref.u), null(Omega), copy(ref.ue), copy(ref.u), T(0.0),
        copy(ref.H_ice), copy(ref.H_water),
        ColumnAnomalies(Omega), copy(ref.b),
        copy(ref.bsl), null(Omega), copy(ref.z_ss),
        copy(ref.V_af), copy(ref.V_pov), copy(ref.V_den),
        copy(ref.maskgrounded), copy(ref.maskocean),
        OceanSurfaceChange(z0 = ref.bsl), 0, 1,
    )
end

#########################################################
# Tools
#########################################################
"""
    FastIsoTools(Omega, c, p)
Return a `struct` containing pre-computed tools to perform forward-stepping of the model.
This includes the Green's functions for the computation of the lithosphere and the SSH
perturbation, plans for FFTs, interpolators of the load and the viscosity over time and
preallocated arrays.
"""
struct FastIsoTools{T<:AbstractFloat, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    viscousconvo::InplaceConvolution{T, C, FP, IP}
    elasticconvo::InplaceConvolution{T, C, FP, IP}
    dz_ssconvo::InplaceConvolution{T, C, FP, IP}
    pfft!::FP
    pifft!::IP
    Hice::Interpolations.Extrapolation{M, 1, Interpolations.GriddedInterpolation{M, 1, Vector{M},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, <:Any}
    bsl::Interpolations.Extrapolation{T, 1, Interpolations.GriddedInterpolation{T, 1, Vector{T},
        Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{T}}}, Gridded{Linear{Throw{OnGrid}}}, <:Any}
    prealloc::PreAllocated{T, M, C}
end

function FastIsoTools(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}},
    bsl_itp; quad_precision::Int = 4,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}

    # Build in-place convolution for viscous response (only used in ELRA)
    L_w = get_flexural_lengthscale(mean(p.litho_rigidity), p.rho_uppermantle, c.g)
    kei = get_kei(Omega, L_w)
    viscousgreen = calc_viscous_green(Omega, mean(p.litho_rigidity), kei, L_w)
    viscousconvo = InplaceConvolution(viscousgreen, Omega.use_cuda)

    # Build in-place convolution to compute elastic response
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elasticgreen = get_elasticgreen(Omega, greenintegrand_function, quad_support, quad_coeffs)
    elasticconvo = InplaceConvolution(elasticgreen, Omega.use_cuda)

    # Build in-place convolution to compute dz_ss response
    dz_ssgreen = get_dz_ssgreen(Omega, c)
    dz_ssconvo = InplaceConvolution(dz_ssgreen, Omega.use_cuda)

    # FFT plans depening on CPU vs. GPU usage
    pfft!, pifft! = choose_fft_plans(Omega.K, Omega.use_cuda)

    # rhog = p.uppermantle_density .* c.g
    Hice = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel); extrapolation_bc=Flat())

    realmatrices = [null(Omega) for _ in eachindex(fieldnames(PreAllocated))[1:end-3]]
    cplxmatrices = [complex.(null(Omega)) for _ in 1:2]
    convo_out = Omega.arraykernel(Matrix{T}(undef, size(dz_ssconvo.Afft)...))
    prealloc = PreAllocated(realmatrices..., cplxmatrices..., convo_out)
    
    return FastIsoTools(viscousconvo, elasticconvo, dz_ssconvo, pfft!, pifft!,
        Hice, bsl_itp, prealloc)
end

#########################################################
# Output
#########################################################

mutable struct NetcdfOutput{T<:AbstractFloat}
    t::Vector{<:AbstractFloat}
    filename::String
    buffer::Matrix{T}
    varsfi3D::Vector{Symbol}
    varnames3D::Vector{String}
    varsfi1D::Vector{Symbol}
    varnames1D::Vector{String}
    computation_time::Float64
end

function NetcdfOutput(Omega::ComputationDomain{T, L, M}, t, filename;
    varsfi3D = interm_varsfi3D,
    varnames3D = interm_varnames3D,
    varlongnames3D = interm_varlongnames3D,
    varunits3D = interm_varunits3D,
    varsfi1D = interm_varsfi1D,
    varnames1D = interm_varnames1D,
    varlongnames1D = interm_varlongnames1D,
    varunits1D = interm_varunits1D,
    Tout = Float32,
) where {T<:AbstractFloat, L, M}

    isfile(filename) && rm(filename)

    xatts = Dict("longname" => "x", "units" => "m")
    yatts = Dict("longname" => "y", "units" => "m")
    tatts = Dict("longname" => "time", "units" => "yr")

    xdim = NcDim("x", Omega.x, xatts)
    ydim = NcDim("y", Omega.y, yatts)
    tdim = NcDim("t", seconds2years.(t), tatts)

    vars = NcVar[]
    for i in eachindex(varnames3D)
        varatts = Dict("longname" => varlongnames3D[i], "units" => varunits3D[i])
        push!(vars, NcVar(varnames3D[i], [xdim, ydim, tdim]; atts = varatts, t = Tout))
    end
    for i in eachindex(varnames1D)
        varatts = Dict("longname" => varlongnames1D[i], "units" => varunits1D[i])
        push!(vars, NcVar(varnames1D[i], [tdim]; atts = varatts, t = Tout))
    end

    if length(filename) > 0
        isfile(filename) && rm(filename)
        NetCDF.create(filename, vars) do nc
            nothing
        end
    end
    
    buffer = Matrix{Tout}(undef, Omega.Nx, Omega.Ny)
    return NetcdfOutput(t, filename, buffer, varsfi3D, varnames3D,
        varsfi1D, varnames1D, 0.0)
end

interm_varsfi3D = [:u, :ue, :dudt, :b, :dz_ss, :z_ss, :maskgrounded, :H_ice,
    :H_water]
interm_varnames3D = ["u", "ue", "dudt", "b", "dz_ss", "z_ss",
    "maskgrounded", "Hice", "Hwater"]
interm_varlongnames3D = ["Viscous displacement", "Elastic displacement",
    "Viscous displacement rate", "Bedrock position", "SSH parturbation",
    "Sea-surface height (SSH)", "Mask for grounded ice", "Ice thickness", "Water depth"]
interm_varunits3D = ["m", "m", "m/yr", "m", "m", "m", "1", "m", "m"]

interm_varsfi1D = [:bsl]
interm_varnames1D = ["bsl"]
interm_varlongnames1D = ["Barystatic sea level"]
interm_varunits1D = ["m"]



abstract type Output end
struct MinimalOutput{T<:AbstractFloat} <: Output
    t::Vector{T}
end

mutable struct SparseOutput{T<:AbstractFloat} <: Output
    t::Vector{T}
    u::Vector{Matrix{T}}
    ue::Vector{Matrix{T}}
end

function SparseOutput(Omega::ComputationDomain{T, L, M}, t_out::Vector{T}) where
    {T<:AbstractFloat, L, M<:KernelMatrix{T}}
    # initialize with placeholders
    placeholder = Array(null(Omega))
    u = [copy(placeholder) for t in t_out]
    ue = [copy(placeholder) for t in t_out]
    return SparseOutput(t_out, u, ue)
end

#########################################################
# Options
#########################################################

"""
    Options

Return a struct containing the options relative to solving a [`FastIsoProblem`](@ref).
"""
Base.@kwdef struct SolverOptions
    deformation_model::Symbol = :lv_elva  # :lv_elva! or :lv_elra! or :lv_elra
    interactive_sealevel::Bool = false
    internal_loadupdate::Bool = true
    internal_bsl_update::Bool = true
    bsl_itp::Any = linear_interpolation([-Inf, Inf], [0.0, 0.0], extrapolation_bc = Flat())
    diffeq::NamedTuple = (alg = Tsit5(), reltol = 1e-3)
    dt_sl::Real = years2seconds(10.0)
    verbose::Bool = false
end

#########################################################
# Problem definition
#########################################################

"""
    FastIsoProblem(Omega, c, p, t_out)
    FastIsoProblem(Omega, c, p, t_out, Hice)
    FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice)

Return a struct containing all the other structs needed for the forward integration of the
model over `Omega::ComputationDomain` with parameters `c::PhysicalConstants` and
`p::LayeredEarth`. The outputs are stored at `t_out::Vector{<:AbstractFloat}`.
"""
struct FastIsoProblem{T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}, O<:Output}
    Omega::ComputationDomain{T, L, M}
    c::PhysicalConstants{T}
    p::LayeredEarth{T, M}
    opts::SolverOptions
    tools::FastIsoTools{T, M, C, FP, IP}
    ref::ReferenceState{T, M}
    now::CurrentState{T, M}
    ncout::NetcdfOutput{<:AbstractFloat}
    out::O
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real};
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Creating some placeholders in case of an external update of the load.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [null(Omega), null(Omega)]
    return FastIsoProblem(Omega, c, p, t_out, t_Hice_snapshots, Hice_snapshots; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    Hice::KernelMatrix{T};
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Constant interpolator in case viscosity is fixed over time.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [Hice, Hice]
    return FastIsoProblem(Omega, c, p, t_out, t_Hice_snapshots, Hice_snapshots; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}};
    opts::SolverOptions = SolverOptions(),
    u_0::KernelMatrix{T} = null(Omega),
    ue_0::KernelMatrix{T} = null(Omega),
    z_ss_0::KernelMatrix{T} = null(Omega),
    b_0::KernelMatrix{T} = null(Omega),
    bsl_itp = linear_interpolation([extrema(t_out)...], [0.0, 0.0]),
    maskactive::BoolMatrix = kernelcollect(Omega.K .< Inf, Omega),
    output_file::String = "",
    output::String = "nothing",
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}

    if !isa(opts.diffeq.alg, OrdinaryDiffEqAlgorithm) && !isa(opts.diffeq.alg, SimpleEuler)
        error("Provided algorithm for solving ODE is not supported.")
    end

    if opts.interactive_sealevel & (sum(maskactive) > 0.6 * Omega.Nx * Omega.Ny)
        error("Mask defining regions of active load must not cover more than 60%"*
            " of the cells when using an interactive sea level.")
    end

    tools = FastIsoTools(Omega, c, p, t_Hice_snapshots, Hice_snapshots, bsl_itp)

    # Initialise the reference state
    H_ice_0, bsl_0 = tools.Hice(t_out[1]), tools.bsl(t_out[1])
    u_0, ue_0, z_ss_0, b_0, H_ice_0 = kernelpromote([u_0, ue_0,
        z_ss_0, b_0, H_ice_0], Omega.arraykernel)

    if Omega.use_cuda
        maskgrounded = get_maskgrounded(H_ice_0, b_0, z_ss_0, c)
        maskocean = get_maskocean(z_ss_0, b_0, maskgrounded)
    else
        maskgrounded = collect(get_maskgrounded(H_ice_0, b_0, z_ss_0, c))
        maskocean = collect(get_maskocean(z_ss_0, b_0, maskgrounded))
    end

    H_water_0 = watercolumn(H_ice_0, maskgrounded, b_0, z_ss_0, c)
    ref = ReferenceState(u_0, ue_0, H_ice_0, H_water_0, b_0, bsl_0, z_ss_0,
        T(0.0), T(0.0), T(0.0), maskgrounded, maskocean, Omega.arraykernel(maskactive))
    now = CurrentState(Omega, ref)
    ncout = NetcdfOutput(Omega, t_out, output_file)

    if output == "sparse"
        out = SparseOutput(Omega, t_out)
    else
        out = MinimalOutput(t_out)
    end
    return FastIsoProblem(Omega, c, p, opts, tools, ref, now, ncout, out)
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