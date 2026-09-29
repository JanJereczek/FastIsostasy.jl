const A_OCEAN_PD = 3.625e14     # Ocean surface (m^2) as in Goelzer (2020, before Eq. (9))

"""
$(TYPEDSIGNATURES)

Define a reference of barystatic sea level and ocean surface area.
Used in all subtypes of [`AbstractBSL`](@ref) to compute the BSL evolution.

# Fields
$(TYPEDFIELDS)

In the constructor, `T` determines the floating point arithmetic used in all
computations, and `itp_kwargs` allows customization of the interpolation.

Example usage:
```julia
ref = ReferenceBSL()              # assume BSL = 0
```

Custom:
```julia
ref = ReferenceBSL(z = 0.1)       # assume BSL = 0.1 m and compute A accordingly
```
"""
struct ReferenceBSL{T<:AbstractFloat,I<:TimeInterpolation0D{T}}
    "the reference BSL (m), which defaults to 0 (reference year 2020)"
    z::T
    "the reference ocean surface area (m²), computed from `z`"
    A::T
    "a vector of BSL values (m) used for interpolation"
    z_vec::Vector{T}
    "a vector of ocean surface area values (m²) used for interpolation"
    A_vec::Vector{T}
    "an interpolator of ocean surface area over BSL"
    A_itp::I
end

function ReferenceBSL(; z = 0, T = Float32, flat_bc = false)
    z_vec, A_vec, _ = load_oceansurface_data(T = T, verbose = false)
    A_unbiased = A_OCEAN_PD ./ A_vec[argmin(abs.(z_vec))] .* A_vec
    A_itp = TimeInterpolation0D(z_vec, T.(A_unbiased), flat_bc = flat_bc)
    A = interpolate(z, A_itp)
    return ReferenceBSL(T(z), T(A), z_vec, A_vec, A_itp)
end

Base.eltype(ref::ReferenceBSL{T}) where {T<:AbstractFloat} = T

"""
$(TYPEDSIGNATURES)

An abstract type to determine how the barystatic sea level (BSL) is updated
over time.

# Available subtypes
- [`InternalBSLUpdate`](@ref)
- [`ExternalBSLUpdate`](@ref)
"""
abstract type AbstractBSLUpdate end

"""
$(TYPEDSIGNATURES)

Update BSL internally by the model based on the change in ice volume.
"""
struct InternalBSLUpdate <: AbstractBSLUpdate end

"""
$(TYPEDSIGNATURES)

Update the BSL is externally, without any internal update.
"""
struct ExternalBSLUpdate <: AbstractBSLUpdate end

"""
$(TYPEDSIGNATURES)

Abstract type to compute the evolution of the barystatic sea level.

# Available subtypes
- [`ConstantBSL`](@ref)
- [`ConstantOceanSurfaceBSL`](@ref)
- [`PiecewiseConstantBSL`](@ref)
- [`PiecewiseLinearOceanSurfaceBSL`](@ref) (requires `using NLsolve`)
- [`ImposedBSL`](@ref)
- [`CombinedBSL`](@ref)

All subtypes implement the `update_bsl!` function:
```julia
T, delta_V, t = Float64, 1.0e9, 0.0             # Example values
bsl = PiecewiseConstantBSL()        # or any other subtype!
update_bsl!(bsl, delta_V, t)
```
"""
abstract type AbstractBSL{T<:AbstractFloat} end

"""
$(TYPEDSIGNATURES)

Assume that the BSL is constant in time.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ConstantBSL{
    T,                      # <: AbstractFloat
    R,                      # <: ReferenceBSL
} <: AbstractBSL{T}
    "the [`ReferenceBSL`](@ref)"
    ref::R
    "the BSL (m), constant in time"
    z::T
    "the ocean surface area (m²), constant in time"
    A::T
end

ConstantBSL(; ref = ReferenceBSL()) = ConstantBSL(ref, ref.z, ref.A)

"""
$(TYPEDSIGNATURES)

Assume that the ocean surface is constant in time and that the BSL evolves
only according to the changes in ice volume covered by the `RegionalDomain`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ConstantOceanSurfaceBSL{T,R<:ReferenceBSL{T}} <: AbstractBSL{T}
    "the [`ReferenceBSL`](@ref)"
    ref::R
    "the BSL (m) at the current time step"
    z::T
    "the ocean surface area (m²), constant in time"
    A::T
end

ConstantOceanSurfaceBSL(; ref = ReferenceBSL()) =
    ConstantOceanSurfaceBSL(ref, ref.z, ref.A)

"""
$(TYPEDSIGNATURES)

Assume that the ocean surface evolves in time according to a piecewise constant function
of the BSL, which evolves in time according to the changes in ice volume covered by the `RegionalDomain`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct PiecewiseConstantBSL{T,R<:ReferenceBSL{T}} <: AbstractBSL{T}
    "the [`ReferenceBSL`](@ref)"
    ref::R
    "the BSL (m) at the current time step"
    z::T
    "the ocean surface area (m²) at the current time step"
    A::T
end

PiecewiseConstantBSL(; ref = ReferenceBSL()) = PiecewiseConstantBSL(ref, ref.z, ref.A)


"""
    PiecewiseLinearOceanSurfaceBSL{T}
    PiecewiseLinearOceanSurfaceBSL(; ref, mcp_opts)

A `mutable struct` that is only available if `using NLsolve`.

# Fields
$(TYPEDFIELDS)

Note that, unlike [`ConstantOceanSurface`](@ref) and [`PiecewiseConstantOceanSurface`](@ref), this will only work if `using NLsolve`.
"""
mutable struct PiecewiseLinearOceanSurfaceBSL{T,R<:ReferenceBSL{T}} <: AbstractBSL{T}
    "the [`ReferenceBSL`](@ref)"
    ref::R
    "the BSL (m) at the current time step"
    z::T
    "the ocean surface area (m²) at the current time step"
    A::T
    "the residual of the nonlinear equation solved numerically"
    residual::T
    """
    the options of the MCP solver, such as `reformulation`, `autodiff`,
    `iterations`, `ftol` and `xtol`
    """
    mcp_opts::NamedTuple
end

"""
$(TYPEDSIGNATURES)

Impose an externally computed BSL, which is internally computed via a time interpolation.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ImposedBSL{T,R<:ReferenceBSL{T}} <: AbstractBSL{T}
    "the [`ReferenceBSL`](@ref)"
    ref::R
    "the BSL (m) at the current time step"
    z::T
    "the times (yr) at which the BSL is imposed"
    t_vec::Vector{T}
    "the imposed BSL values (m) at `t_vec`"
    z_vec::Vector{T}
    "the interpolation of `z_vec` over `t_vec`"
    z_itp::TimeInterpolation0D{T}
end

"""
$(TYPEDSIGNATURES)

This imposes a mixture of BSL. For instance, if you simulate Antarctica over the LGM,
you can impose an offline BSL contribution from the other ice sheets via `bsl1`. The
contribution of Antarctica will be intercatively added to this via `bsl2`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct CombinedBSL{T,B1<:ImposedBSL,B2<:AbstractBSL} <: AbstractBSL{T}
    "the imposed, offline BSL contribution"
    bsl1::B1
    "the interactively computed BSL contribution"
    bsl2::B2
end

CombinedBSL(bsl1::ImposedBSL{T}, bsl2::AbstractBSL{T}) where {T} =
    CombinedBSL{T,typeof(bsl1),typeof(bsl2)}(bsl1, bsl2)

"""
$(TYPEDSIGNATURES)

Update the BSL and ocean surface based on the input `delta_V` (in m^3) and
on a subtype of [`AbstractBSL`](@ref).
"""
function update_bsl!(bsl::ConstantBSL, delta_V, t)
    bsl.z = bsl.ref.z
    return nothing
end

function update_bsl!(bsl::ConstantOceanSurfaceBSL, delta_V, t)
    bsl.A = bsl.ref.A
    bsl.z += delta_V / bsl.A
    return nothing
end

function update_bsl!(bsl::PiecewiseConstantBSL, delta_V, t)
    bsl.z += delta_V / bsl.A
    bsl.A = interpolate(bsl.z, bsl.ref.A_itp)
    return nothing
end

function update_bsl!(bsl::ImposedBSL, delta_V, t)
    bsl.z = interpolate(t, bsl.z_itp)
    return nothing
end