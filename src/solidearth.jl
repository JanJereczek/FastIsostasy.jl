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

Note that none of these subtypes carries the unrelaxed elastic response: that is
computed separately by the Farrell Green's-function convolution, which dispatches
on [`AbstractLithosphere`](@ref).

Available subtypes are:
- [`RigidMantle`](@ref)
- [`RelaxedMantle`](@ref)
- [`ViscousMantle`](@ref)
- [`TransientCreepMantle`](@ref)
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
    TransientCreepMantle(; shearmodulus, relaxation_strength, kelvin_time)

Transient (primary) creep on top of the steady creep of [`ViscousMantle`](@ref):
the steady Maxwell dashpot (viscosity `η₁`, taken from `SolidEarth`) in series
with `N` Kelvin-Voigt branches forming a Prony series, so that the total viscous
displacement splits as `u = u_M + Σⱼ u_K[j]`. `N = 1` is the classic Burgers
body; `N ≈ 3-5` log-spaced branches approximate the extended Burgers /
Faul-Jackson relaxation spectrum of Ivins & Caron (2021).

Named for the phenomenon rather than for one rheological body, because the
`N`-branch Prony form is the general case and Burgers is only its `N = 1` member.

Note that, unlike [`ViscousMantle`](@ref), this model *does* carry shear moduli:
each branch has `μ₂ⱼ = shearmodulus / Δⱼ` and `η₂ⱼ = τⱼ μ₂ⱼ`. These are internal
to the transient band and remain distinct from the unrelaxed elastic response,
which the Farrell convolution keeps handling as for every other mantle. It
*extends* `ViscousMantle` rather than replacing it: as `Δⱼ → 0` the Kelvin
branches lock (`u_K → 0`) and the model reduces to it exactly.

Scalar (laterally constant) parameters only, and supported solely on the
semi-implicit path — [`LaterallyConstantLithosphere`](@ref) or
[`RigidLithosphere`](@ref) with [`ComplexFFTBackend`](@ref) and a fixed step
(`SolverOptions(integ = EulerIntegrator(dt = ...))`).

# Fields
$(TYPEDFIELDS)

# Example
Classic Burgers body with the Ivins & Caron (2021) Fig. 8 parameters:
```jldoctest
julia> using FastIsostasy

julia> m = TransientCreepMantle(shearmodulus = 67e9, relaxation_strength = 1.2,
           kelvin_time = 7.14);

julia> FastIsostasy.nbranches(m)
1
```

See `roadmaps/burgers.md` for the design and derivation.
"""
struct TransientCreepMantle{T<:AbstractFloat,N} <: AbstractMantle
    "unrelaxed shear modulus `μ₁` of the mantle [Pa]"
    shearmodulus::T
    "relaxation strength `Δⱼ = μ₁/μ₂ⱼ` of each Kelvin branch"
    relaxation_strength::NTuple{N,T}
    "retardation time `τⱼ = η₂ⱼ/μ₂ⱼ` of each Kelvin branch [yr]"
    kelvin_time::NTuple{N,T}
end

function TransientCreepMantle(; shearmodulus, relaxation_strength, kelvin_time)
    Δ, τ = _branch_tuple(relaxation_strength), _branch_tuple(kelvin_time)
    length(Δ) == length(τ) || throw(DimensionMismatch(
        "relaxation_strength has $(length(Δ)) branch(es) but kelvin_time has " *
        "$(length(τ)); a Prony series needs one Δⱼ per τⱼ."))
    all(>(0), Δ) || throw(ArgumentError(
        "every relaxation strength Δⱼ must be > 0 (got $Δ). Δ → 0 is the " *
        "ViscousMantle limit — use that type instead."))
    all(>(0), τ) || throw(ArgumentError(
        "every retardation time τⱼ must be > 0 (got $τ)."))
    shearmodulus > 0 || throw(ArgumentError("shearmodulus must be > 0."))
    T = float(promote_type(typeof(shearmodulus), eltype(Δ), eltype(τ)))
    N = length(Δ)
    return TransientCreepMantle{T,N}(
        T(shearmodulus),
        NTuple{N,T}(Δ),
        NTuple{N,T}(τ),
    )
end

_branch_tuple(x::Real) = (x,)
_branch_tuple(x) = Tuple(x)

"""
$(TYPEDSIGNATURES)

Classic Burgers body: the `N = 1` member of [`TransientCreepMantle`](@ref), a
Maxwell dashpot in series with a single Kelvin-Voigt element. Thin naming
convenience over `TransientCreepMantle(; shearmodulus, relaxation_strength,
kelvin_time)` for when `(Δ, τ)` are already known (e.g. from a published fit),
as opposed to [`ExtendedBurgersMantle`](@ref), which fits them from a
continuous relaxation-time spectrum.
"""
BurgersMantle(; shearmodulus, relaxation_strength, kelvin_time) =
    TransientCreepMantle(; shearmodulus, relaxation_strength, kelvin_time)

"""
$(TYPEDSIGNATURES)

Extended Burgers mantle: a [`TransientCreepMantle`](@ref) with `nbranches`
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
    return TransientCreepMantle(; shearmodulus, relaxation_strength = Δ,
        kelvin_time = τ)
end

"""
$(TYPEDSIGNATURES)

Number of Kelvin branches carried by a mantle rheology, i.e. how many transient
displacement fields `sim.now.u_K` must hold. Zero for every rheology whose creep
is purely steady-state.
"""
nbranches(::AbstractMantle) = 0
nbranches(::TransientCreepMantle{T,N}) where {T,N} = N

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

which initializes a lithosphere of thickness ``T_1 = 100 \\mathrm{km}``, a viscous
channel between ``T_1``and ``T_2 = 300 \\mathrm{km}``and a viscous halfspace starting
at ``T_2``. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `domain::RegionalDomain`.
"""
mutable struct SolidEarth{
    T,  # <:AbstractFloat,
    M,  # <:AbstractMatrix{T},
    B,  # <:AbstractMatrix{Bool},
    LI, # <:AbstractLithosphere,
    MA, # <:AbstractMantle,
    CA, # <:AbstractCalibration,
    CO, # <:AbstractCompressibility,
    LU, # <:AbstractLumping,
    LC, # <:AbstractLithosphereColumn,
}
    lithosphere::LI
    mantle::MA
    calibration::CA
    compressibility::CO
    lumping::LU
    lithosphere_column::LC
    effective_viscosity::M
    pseudodiff_scaling::M
    scaled_pseudodiff_inv::M
    litho_thickness::M
    litho_rigidity::M
    maskactive::B
    litho_poissonratio::T
    mantle_poissonratio::T
    tau::M
    scale_elralength::T
    litho_youngmodulus::T
    litho_shearmodulus::T
    rho_uppermantle::T
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
    padlen = maximum(length(d[1]) for d in descriptors) + 2
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end
