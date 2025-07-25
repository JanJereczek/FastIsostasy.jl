const A_OCEAN_PD = 3.625e14     # Ocean surface (m^2) as in Goelzer (2020, before Eq. (9))

"""
    ReferenceBSL(z = 0; T = Float32, itp_kwargs = (extrapolation_bc = Flat()))

A `struct` used in all subtypes of `AbstractBSL` to define a reference of barystatic
sea level and ocean surface area. Contains:
- `z`: the reference BSL (m), which defaults to 0 (reference year 2020).
- `A`: the reference ocean surface area (m^2) computed based on `z`.
- `z_vec`: a vector of BSL values (m) used for interpolation.
- `A_vec`: a vector of ocean surface area values (m^2) used for interpolation.
- `A_itp`: an interpolator function for ocean surface area over BSL.

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
struct ReferenceBSL{T<:AbstractFloat, I<:TimeInterpolation0D{T}}
    z::T
    A::T
    z_vec::Vector{T}
    A_vec::Vector{T}
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

abstract type AbstractUpdateBSL end
struct InternalUpdateBSL <: AbstractUpdateBSL end
struct ExternalUpdateBSL <: AbstractUpdateBSL end

"""
    AbstractBSL

Abstract type to compute the evolution of the barystatic sea level.
Available subtypes are:
- [`ConstantBSL`](@ref)
- [`ConstantOceanSurfaceBSL`](@ref)
- [`PiecewiseConstantBSL`](@ref)
- [`PiecewiseLinearBSL`](@ref)
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
    ConstantBSL

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceBSL`](@ref).
- `z`: the BSL, considered constant in time.
- `A`: the ocean surface area, considered constant in time.

Assume that the BSL is constant in time.
"""
mutable struct ConstantBSL{
    T,                      # <: AbstractFloat
    R,                      # <: ReferenceBSL
} <: AbstractBSL{T}

    ref::R
    z::T
    A::T
end

ConstantBSL(; ref = ReferenceBSL()) = ConstantBSL(ref, ref.z, ref.A)

"""
    ConstantOceanSurfaceBSL{T}

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceBSL`](@ref).
- `z`: the BSL at current time step.
- `A`: the ocean surface area, considered constant in time.

Assume that the ocean surface is constant in time and that the BSL evolves
only according to the changes in ice volume covered by the `RegionalDomain`.
"""
mutable struct ConstantOceanSurfaceBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    A::T
end

ConstantOceanSurfaceBSL(; ref = ReferenceBSL()) = ConstantOceanSurfaceBSL(ref, ref.z, ref.A)

"""
    PiecewiseConstantBSL{T}

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceBSL`](@ref).
- `z`: the BSL at current time step.
- `A`: the ocean surface at current time step.

Assume that the ocean surface evolves in time according to a piecewise constant function
of the BSL, which evolves in time according to the changes in ice volume covered by the `RegionalDomain`.
"""
mutable struct PiecewiseConstantBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    A::T
end

PiecewiseConstantBSL(; ref = ReferenceBSL()) = 
    PiecewiseConstantBSL(ref, ref.z, ref.A)

"""
    ImposedBSL

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceBSL`](@ref).
- `z`: the BSL at current time step.
- `t_vec`: the time vector.
- `z_vec`: the BSL values corresponding to the time vector.
- `z_itp`: an interpolation of `z_vec` over `t_vec`.

Impose an externally computed BSL, which is internally computed via a time interpolation.
"""
mutable struct ImposedBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    t_vec::Vector{T}
    z_vec::Vector{T}
    z_itp::TimeInterpolation0D{T}
end

"""
    CombinedBSL

A `mutable struct`containing:
- `bsl1`: an [`ImposedBSL`](@ref).
- `bsl2`: an [`AbstractBSL`](@ref).

This imposes a mixture of BSL. For instance, if you simulate Antarctica over the LGM,
you can impose an offline BSL contribution from the other ice sheets via `bsl1`. The
contribution of Antarctica will be intercatively added to this via `bsl2`.
"""
mutable struct CombinedBSL{T, B1<:ImposedBSL, B2<:AbstractBSL} <: AbstractBSL{T}
    bsl1::B1
    bsl2::B2
end

"""
    update_bsl!(bsl::AbstractBSL, delta_V, t)

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