##############################################################
# Lithosphere
##############################################################

"""
$(TYPEDSIGNATURES)

Available subtypes are:
- [`RigidLithosphere`](@ref)
- [`LaterallyConstantLithosphere`](@ref)
- [`LaterallyVariableLithosphere`](@ref)
"""
abstract type AbstractLithosphere end

"""
$(TYPEDSIGNATURES)

Assume a rigid lithosphere, i.e. the elastic deformation is neglected.
"""
struct RigidLithosphere <: AbstractLithosphere end

"""
$(TYPEDSIGNATURES)

Assume a laterally constant lithospheric thickness (and rigidity) across the domain.
This generally improves the performance of the solver, but is less realistic.
"""
struct LaterallyConstantLithosphere <: AbstractLithosphere end

"""
$(TYPEDSIGNATURES)

Assume a laterally variable lithospheric thickness (and rigidity) across the domain.
This generally improves the realism of the model, but is more computationally expensive.
"""
struct LaterallyVariableLithosphere <: AbstractLithosphere end

##############################################################
# Mantle
##############################################################

"""
$(TYPEDSIGNATURES)

The rheology of the mantle. This axis describes *what is modelled*; how the
spectral step is computed is the orthogonal [`AbstractFFTBackend`](@ref) axis.

Most subtypes carry no unrelaxed elastic response: that is computed separately by
the Farrell Green's-function convolution, which dispatches on
[`AbstractLithosphere`](@ref). The two `*ViscoElastic*` subtypes are the
exception. They put the elastic spring in series with the viscous elements of
a homogeneous half-space, so that elastic and viscous deformation are coupled.

Available subtypes are:
- [`RigidMantle`](@ref)
- [`RelaxedMantle`](@ref)
- [`ViscousMantle`](@ref)
- [`TransientViscousMantle`](@ref)
- [`ViscoElasticMantle`](@ref)
- [`TransientViscoElasticMantle`](@ref)
"""
abstract type AbstractMantle end

"""
$(TYPEDSIGNATURES)

Assume a rigid mantle that does not deform.
"""
struct RigidMantle <: AbstractMantle end

"""
$(TYPEDSIGNATURES)

Assume a relaxed mantle that deforms according to a relaxation time.
This is generally less realistic and offers worse performance than a viscous mantle.
It is only included for legacy purpose (e.g. comparison among solvers).
"""
struct RelaxedMantle <: AbstractMantle end

"""
$(TYPEDSIGNATURES)

Assume a viscous mantle that deforms according to a viscosity, i.e. steady-state
(secondary) creep only.

The name is deliberately *not* `MaxwellMantle`: a Maxwell body is a spring in
series with a dashpot, but the spring — the unrelaxed elastic response — is not
handled here. It is computed independently by the Farrell Green's-function
convolution in `update_elasticresponse!`, which dispatches on
[`AbstractLithosphere`](@ref) and can be switched off entirely
([`RigidLithosphere`](@ref)). This type supplies the viscous half-space only.

Whether the spectral step uses complex or half-spectrum FFTs is a separate,
orthogonal choice — see [`AbstractFFTBackend`](@ref) and the `fft` field of
[`SolverOptions`](@ref).

This is the most realistic mantle model with a steady creep law and generally
offers the best performance. It is the default mantle model used in the solver.
"""
struct ViscousMantle <: AbstractMantle end

"""
    TransientViscousMantle(; shearmodulus, relaxation_strength, kelvin_time)

Transient (primary) creep on top of the steady creep of [`ViscousMantle`](@ref):
the steady Maxwell dashpot (viscosity `η₁`, taken from `SolidEarth`) in series
with `N` Kelvin-Voigt branches forming a Prony series, so that the total viscous
displacement splits as `u = u_M + Σⱼ u_K[j]`. `N = 1` is the classic Burgers
body. `N ≈ 6` log-spaced branches approximate the extended Burgers /
Faul-Jackson relaxation spectrum of Ivins & Caron (2021), see
[`ExtendedBurgersMantle`](@ref).

Each branch has `μ₂ⱼ = shearmodulus / Δⱼ` and `η₂ⱼ = τⱼ μ₂ⱼ`. The unrelaxed
elastic spring `μ₁ = shearmodulus` itself is *not* in series with these
branches: as for [`ViscousMantle`](@ref), the elastic response is computed apart
by the Farrell convolution and added to the viscous one. That is the difference
with [`TransientViscoElasticMantle`](@ref), which couples the spring in. As
`Δⱼ → 0` the Kelvin branches lock (`u_K → 0`) and the model reduces to
`ViscousMantle` exactly.

Scalar (laterally constant) parameters only, and supported solely on the
semi-implicit path: [`LaterallyConstantLithosphere`](@ref) or
[`RigidLithosphere`](@ref) with [`ComplexFFTBackend`](@ref) and a fixed step
(`SolverOptions(integ = EulerIntegrator(dt = ...))`).

# Fields
$(TYPEDFIELDS)

# Example
Classic Burgers body with the Ivins & Caron (2021) Fig. 8 parameters:
```jldoctest
julia> using FastIsostasy

julia> m = TransientViscousMantle(shearmodulus = 67e9, relaxation_strength = 1.2,
           kelvin_time = 7.14);

julia> FastIsostasy.nbranches(m)
1
```

See `fastisostasy-roadmap/burgers.md` for the design and derivation.
"""
struct TransientViscousMantle{T<:AbstractFloat,N} <: AbstractMantle
    "unrelaxed shear modulus `μ₁` of the mantle [Pa]"
    shearmodulus::T
    "relaxation strength `Δⱼ = μ₁/μ₂ⱼ` of each Kelvin branch"
    relaxation_strength::NTuple{N,T}
    "retardation time `τⱼ = η₂ⱼ/μ₂ⱼ` of each Kelvin branch [yr]"
    kelvin_time::NTuple{N,T}
end

TransientViscousMantle(; shearmodulus, relaxation_strength, kelvin_time) =
    TransientViscousMantle(_prony_params(shearmodulus, relaxation_strength,
        kelvin_time)...)

"""
$(TYPEDSIGNATURES)

Maxwell mantle: the elastic spring (`shearmodulus`, `μ`) in series with the
viscous dashpot (viscosity `η`, taken from `SolidEarth`) of a homogeneous,
incompressible half-space. Per Fourier mode of wavenumber `k`, the vertical
response to a surface load is `1/(β + 2k μ̃(s))` with `1/μ̃(s) = 1/μ + 1/(ηs)`
and `β = ρg + Dk⁴`.

Unlike [`ViscousMantle`](@ref), where the elastic displacement comes from the
Farrell convolution and is simply added to the viscous one, the spring here is
coupled to the dashpot: the elastic displacement `ue` relaxes as the viscous one
grows, and the viscous relaxation time becomes `η/μ + 2ηk/β` instead of `2ηk/β`.
`ue` is therefore computed by the mantle, every time step, and the Farrell
convolution is bypassed. For the same reason the load does not include the
lithospheric column anomaly `ρ_litho · ue`, whose buoyancy is already part of `β`.

Scalar (laterally constant) parameters only, on the same semi-implicit path as
[`TransientViscousMantle`](@ref).

# Fields
$(TYPEDFIELDS)

# Example
```jldoctest
julia> using FastIsostasy

julia> m = ViscoElasticMantle(shearmodulus = 67e9);

julia> FastIsostasy.nbranches(m)
0
```
"""
Base.@kwdef struct ViscoElasticMantle{T<:AbstractFloat} <: AbstractMantle
    "unrelaxed shear modulus `μ` of the mantle [Pa]"
    shearmodulus::T
end

"""
    TransientViscoElasticMantle(; shearmodulus, relaxation_strength, kelvin_time)
    TransientViscoElasticMantle(mantle::TransientViscousMantle)

Transient creep with the elastic spring coupled in: the viscoelastic
counterpart of [`TransientViscousMantle`](@ref), exactly as
[`ViscoElasticMantle`](@ref) is that of [`ViscousMantle`](@ref). The spring
(`μ₁ = shearmodulus`), the Maxwell dashpot (`η₁`, from `SolidEarth`) and the `N`
Kelvin-Voigt branches (`μ₂ⱼ = μ₁/Δⱼ`, `η₂ⱼ = τⱼ μ₂ⱼ`) are all in series, i.e.
the creep function is

    J(t) = [1 + t/τ_M + Σⱼ Δⱼ (1 − exp(−t/τⱼ))] / μ₁,   τ_M = η₁/μ₁

which is the Prony-series form of the extended Burgers model of Ivins & Caron
(2021). The elastic displacement `ue` is computed by the mantle, as described in
[`ViscoElasticMantle`](@ref).

The second method converts a [`TransientViscousMantle`](@ref), e.g. one built by
[`ExtendedBurgersMantle`](@ref), keeping its parameters.

# Fields
$(TYPEDFIELDS)

# Example
```jldoctest
julia> using FastIsostasy

julia> m = TransientViscoElasticMantle(ExtendedBurgersMantle(shearmodulus = 67e9,
           relaxation_strength = 1.2, alpha = 0.5, tau_L = 1e-4, tau_H = 7.14,
           nbranches = 6));

julia> FastIsostasy.nbranches(m)
6
```
"""
struct TransientViscoElasticMantle{T<:AbstractFloat,N} <: AbstractMantle
    "unrelaxed shear modulus `μ₁` of the mantle [Pa]"
    shearmodulus::T
    "relaxation strength `Δⱼ = μ₁/μ₂ⱼ` of each Kelvin branch"
    relaxation_strength::NTuple{N,T}
    "retardation time `τⱼ = η₂ⱼ/μ₂ⱼ` of each Kelvin branch [yr]"
    kelvin_time::NTuple{N,T}
end

TransientViscoElasticMantle(; shearmodulus, relaxation_strength, kelvin_time) =
    TransientViscoElasticMantle(_prony_params(shearmodulus, relaxation_strength,
        kelvin_time)...)
TransientViscoElasticMantle(m::TransientViscousMantle) =
    TransientViscoElasticMantle(m.shearmodulus, m.relaxation_strength, m.kelvin_time)

function _prony_params(shearmodulus, relaxation_strength, kelvin_time)
    Δ, τ = _branch_tuple(relaxation_strength), _branch_tuple(kelvin_time)
    length(Δ) == length(τ) || throw(DimensionMismatch(
        "relaxation_strength has $(length(Δ)) branch(es) but kelvin_time has " *
        "$(length(τ)); a Prony series needs one Δⱼ per τⱼ."))
    all(>(0), Δ) || throw(ArgumentError(
        "every relaxation strength Δⱼ must be > 0 (got $Δ). Δ → 0 is the " *
        "ViscousMantle (or ViscoElasticMantle) limit — use that type instead."))
    all(>(0), τ) || throw(ArgumentError(
        "every retardation time τⱼ must be > 0 (got $τ)."))
    shearmodulus > 0 || throw(ArgumentError("shearmodulus must be > 0."))
    T = float(promote_type(typeof(shearmodulus), eltype(Δ), eltype(τ)))
    N = length(Δ)
    return T(shearmodulus), NTuple{N,T}(Δ), NTuple{N,T}(τ)
end

# Mantles solved by the coupled (N+1)-field semi-implicit spectral step, and
# the subset of them whose elastic spring is coupled into that step.
const ElasticSpringMantle = Union{ViscoElasticMantle,TransientViscoElasticMantle}
const SpectralCreepMantle = Union{TransientViscousMantle,ElasticSpringMantle}

# Kelvin branches as (Δⱼ, τⱼ) pairs; empty for a pure Maxwell body.
kelvin_branches(m::Union{TransientViscousMantle,TransientViscoElasticMantle}) =
    zip(m.relaxation_strength, m.kelvin_time)
kelvin_branches(::ViscoElasticMantle{T}) where {T} = zip((), ())

_branch_tuple(x::Real) = (x,)
_branch_tuple(x) = Tuple(x)

"""
$(TYPEDSIGNATURES)

Classic Burgers body: the `N = 1` member of [`TransientViscousMantle`](@ref), a
Maxwell dashpot in series with a single Kelvin-Voigt element. Thin naming
convenience over `TransientViscousMantle(; shearmodulus, relaxation_strength,
kelvin_time)` for when `(Δ, τ)` are already known (e.g. from a published fit),
as opposed to [`ExtendedBurgersMantle`](@ref), which fits them from a
continuous relaxation-time spectrum.
"""
BurgersMantle(; shearmodulus, relaxation_strength, kelvin_time) =
    TransientViscousMantle(; shearmodulus, relaxation_strength, kelvin_time)

"""
$(TYPEDSIGNATURES)

Extended Burgers mantle: a [`TransientViscousMantle`](@ref) with `nbranches`
Kelvin branches fit, via [`fit_prony_series`](@ref), to the continuous
Faul-Jackson absorption-band spectrum of the extended Burgers model (EBM) of
Ivins & Caron (2021). See [`fit_prony_series`](@ref) for the fitting procedure
and its `fit_error`; this constructor discards `fit_error` — call
`fit_prony_series` directly first to check it before committing to
`nbranches`.
"""
function ExtendedBurgersMantle(; shearmodulus, relaxation_strength, alpha,
        tau_L, tau_H, nbranches)
    Δ, τ, _ = fit_prony_series(; relaxation_strength, alpha, tau_L, tau_H,
        nbranches)
    return TransientViscousMantle(; shearmodulus, relaxation_strength = Δ,
        kelvin_time = τ)
end

"""
$(TYPEDSIGNATURES)

Number of Kelvin branches carried by a mantle rheology, i.e. how many transient
displacement fields `sim.now.u_K` must hold. Zero for every rheology whose creep
is purely steady-state.
"""
nbranches(::AbstractMantle) = 0
nbranches(::TransientViscousMantle{T,N}) where {T,N} = N
nbranches(::TransientViscoElasticMantle{T,N}) where {T,N} = N

# Number of complex buffers the coupled elastic spring needs (see `PreAllocated`).
nsprings(::AbstractMantle) = 0
nsprings(::ElasticSpringMantle) = 1

################################################################
# Lithosphere behaviour in column anomaly
################################################################

"""
$(TYPEDSIGNATURES)

An abstract type for lithosphere column behaviour. Used to multiple dispatch the computation of the mass anomaly. Available subtypes are:
 - [`CompressibleLithosphereColumn`](@ref)
 - [`IncompressibleLithosphereColumn`](@ref)
"""
abstract type AbstractLithosphereColumn end

"""
$(TYPEDSIGNATURES)

A subtype of [`AbstractLithosphereColumn`](@ref) to assume that the lithosphere does not contribute to gravity anomalies due to compression.
"""
struct CompressibleLithosphereColumn <: AbstractLithosphereColumn end

"""
$(TYPEDSIGNATURES)

A subtype of [`AbstractLithosphereColumn`](@ref) to assume that the lithosphere contributes to gravity anomalies due to incompressiblity (and therefore flow within the lithosphere).
"""
struct IncompressibleLithosphereColumn <: AbstractLithosphereColumn end

###############################################################
# Solid Earth
###############################################################

const DEFAULT_RHO_LITHO = 3.2e3
const DEFAULT_LITHO_YOUNGMODULUS = 6.6e10
const DEFAULT_LITHO_POISSONRATIO = 0.28
const DEFAULT_LITHO_THICKNESS = 88e3
const DEFAULT_RHO_UPPERMANTLE = 3.4e3
const DEFAULT_MANTLE_POISSONRATIO = 0.28
const DEFAULT_MANTLE_TAU = 855.0

"""
$(TYPEDSIGNATURES)

Return a struct containing all information related to the lateral variability of
solid-Earth parameters. To initialize with values other than default, run:

```julia
domain = RegionalDomain(3000e3, 7)
lb = [100e3, 300e3]
lv = [1e19, 1e21]
solidearth = SolidEarth(domain, layer_boundaries = lb, layer_viscosities = lv)
```

which initializes a lithosphere of thickness `T₁ = 100 km`, a viscous
channel between `T₁` and `T₂ = 300 km` and a viscous halfspace starting
at `T₂`. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `domain::RegionalDomain`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct SolidEarth{
    T,  # <:AbstractFloat,
    M,  # <:AbstractMatrix{T},
    B,  # <:AbstractMatrix{Bool},
    LI, # <:AbstractLithosphere,
    MA, # <:AbstractMantle,
    CA, # <:AbstractCalibration,
    CO, # <:AbstractCompressibility,
    LU, # <:AbstractViscosityLumping,
    LC, # <:AbstractLithosphereColumn,
}
    "the [`AbstractLithosphere`](@ref) defining the rheology of the lithosphere"
    lithosphere::LI
    "the [`AbstractMantle`](@ref) defining the rheology of the mantle"
    mantle::MA
    "the [`AbstractCalibration`](@ref) applied to the effective viscosity"
    calibration::CA
    "the `AbstractCompressibility` applied to the effective viscosity"
    compressibility::CO
    "the [`AbstractViscosityLumping`](@ref) lumping the layered viscosity into an effective one"
    lumping::LU
    "the `AbstractLithosphereColumn` deciding how the lithosphere contributes to the column anomaly"
    lithosphere_column::LC
    "the effective mantle viscosity (Pa s)"
    effective_viscosity::M
    "the scaling of the pseudo-differential operator resulting from the viscosity lumping (1)"
    pseudodiff_scaling::M
    "the inverse of the scaled pseudo-differential operator, precomputed for the viscous update"
    scaled_pseudodiff_inv::M
    "the lithospheric thickness (m)"
    litho_thickness::M
    "the flexural rigidity of the lithosphere (N m)"
    litho_rigidity::M
    "the mask of the cells where the load is active"
    maskactive::B
    "the Poisson ratio of the lithosphere (1)"
    litho_poissonratio::T
    "the Poisson ratio of the mantle (1)"
    mantle_poissonratio::T
    "the relaxation time of the mantle (yr), used by the [`RelaxedMantle`](@ref)"
    tau::M
    "the scaling of the ELRA length, following LeMeur (1996, text below Eq. 3) (1)"
    scale_elralength::T
    "the Young modulus of the lithosphere (N m⁻²)"
    litho_youngmodulus::T
    "the shear modulus of the lithosphere (N m⁻²)"
    litho_shearmodulus::T
    "the mean density of the topmost upper mantle (kg m⁻³)"
    rho_uppermantle::T
    "the mean density of the lithosphere (kg m⁻³)"
    rho_litho::T
end

function SolidEarth(
    domain::RegionalDomain{T,L,M};
    lithosphere = LaterallyVariableLithosphere(),
    mantle = ViscousMantle(),
    calibration = NoCalibration(),
    compressibility = CompressibleMantle(),
    lumping = FreqDomainViscosityLumping(),
    lithosphere_column = IncompressibleLithosphereColumn(),
    maskactive = domain.R .< Inf,
    layer_boundaries = T.([88e3, 400e3]),
    layer_viscosities = T.([1e19, 1e21]),           # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
    litho_youngmodulus = T(DEFAULT_LITHO_YOUNGMODULUS),              # (N/m^2)
    litho_poissonratio = T(DEFAULT_LITHO_POISSONRATIO),
    mantle_poissonratio = T(DEFAULT_MANTLE_POISSONRATIO),
    tau = T(DEFAULT_MANTLE_TAU),
    scale_elralength = T(1),                        # Following LeMeur (1996, text below Eq. 3)
    rho_uppermantle = T(DEFAULT_RHO_UPPERMANTLE),   # Mean density of topmost upper mantle (kg m^-3)
    rho_litho = T(DEFAULT_RHO_LITHO),               # Mean density of lithosphere (kg m^-3)
) where {T<:AbstractFloat,L,M}

    if tau isa Real
        tau = fill(tau, domain)
    end
    tau = kernelpromote(tau, domain.backend)

    if layer_boundaries isa Vector
        layer_boundaries = matrify(layer_boundaries, domain.nx, domain.ny)
    end

    if layer_viscosities isa Vector
        layer_viscosities = matrify(layer_viscosities, domain.nx, domain.ny)
    end

    litho_thickness = zeros(T, domain.nx, domain.ny)
    litho_thickness .= view(layer_boundaries, :, :, 1)

    litho_rigidity =
        get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)
    effective_viscosity, pseudodiff_scaling = get_effective_viscosity_and_scaling(
        domain,
        layer_viscosities,
        layer_boundaries,
        maskactive,
        lumping,
    )

    apply_compressibility!(effective_viscosity, mantle_poissonratio, compressibility)
    apply_calibration!(effective_viscosity, calibration)

    litho_thickness,
    litho_rigidity,
    effective_viscosity,
    pseudodiff_scaling,
    maskactive = kernelpromote(
        [
            litho_thickness,
            litho_rigidity,
            effective_viscosity,
            pseudodiff_scaling,
            maskactive,
        ],
        domain.backend,
    )

    scaled_pseudodiff_inv = 1 ./ (pseudodiff_scaling .* domain.pseudodiff)

    litho_shearmodulus = get_shearmodulus(litho_youngmodulus, litho_poissonratio)

    return SolidEarth(
        lithosphere,
        mantle,
        calibration,
        compressibility,
        lumping,
        lithosphere_column,
        effective_viscosity,
        pseudodiff_scaling,
        scaled_pseudodiff_inv,
        litho_thickness,
        litho_rigidity,
        kernelcollect(maskactive, domain),
        litho_poissonratio,
        mantle_poissonratio,
        tau,
        scale_elralength,
        litho_youngmodulus,
        litho_shearmodulus,
        rho_uppermantle,
        rho_litho,
    )

end

function Base.show(io::IO, ::MIME"text/plain", se::SolidEarth)
    descriptors = [
        "Lithosphere" => typeof(se.lithosphere),
        "Mantle" => typeof(se.mantle),
        "Calibration" => typeof(se.calibration),
        "Compressibility" => typeof(se.compressibility),
        "Viscosity lumping" => typeof(se.lumping),
        "Lithosphere column" => typeof(se.lithosphere_column),
        "extrema(effective_viscosity)" => extrema(se.effective_viscosity),
        "extrema(litho_thickness)" => extrema(se.litho_thickness),
        "active cells" => "$(sum(se.maskactive)) / $(length(se.maskactive))",
        "litho_youngmodulus" => se.litho_youngmodulus,
        "litho_poissonratio, mantle_poissonratio" =>
            [se.litho_poissonratio, se.mantle_poissonratio],
        "rho_uppermantle, rho_litho" => [se.rho_uppermantle, se.rho_litho],
    ]
    show_descriptors(io, descriptors)
end
