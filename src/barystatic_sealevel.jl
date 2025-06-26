const A_OCEAN_PD = 3.625e14     # Ocean surface (m^2) as in Goelzer (2020, before Eq. (9))

"""
    ReferenceBSL{T<:AbstractFloat, I<:TimeInterpolation0D}
    ReferenceBSL(z = 0; T = Float32, itp_kwargs = (extrapolation_bc = Flat()))

A `struct` used in all subtypes of `AbstractBSL` to define a fixed
reference of barystatic sea level and ocean surface area. Contains:
- `z`: the reference BSL (m), which defaults to 0 (reference year 2020).
- `A`: the reference ocean surface area (m^2).
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

function ReferenceBSL(z; T = Float32, flat_bc = false)
    z_vec, A_vec = load_oceansurface_data(T = T, verbose = false)
    A_unbiased = A_OCEAN_PD ./ A_vec[argmin(abs.(z_vec))] .* A_vec
    A_itp = TimeInterpolation0D(z_vec, T.(A_unbiased), flat_bc = flat_bc)
    A = interpolate!(z, A_itp)
    return ReferenceBSL(T(z), T(A), z_vec, A_vec, A_itp)
end

ReferenceBSL() = ReferenceBSL(0)

Base.eltype(ref::ReferenceBSL{T}) where {T<:AbstractFloat} = T

"""
    AbstractBSL

Abstract type for ocean surface. Available subtypes are:
- [`ConstantBSL`](@ref)
- [`ConstantOceanSurfaceBSL`](@ref)
- [`PiecewiseConstantOceanSurfaceBSL`](@ref)
- [`PiecewiseLinearOceanSurface`](@ref)
- [`ExternalBSL`](@ref)
- [`CombinedBSL`](@ref)

All subtypes implement the `update_bsl!` function:
```julia
T, delta_V = Float64, 1.0e9                 # Example values
bsl = PiecewiseConstantOceanSurfaceBSL{T}()     # or any other subtype!
update_bsl!(bsl, delta_V)
```
"""
abstract type AbstractBSL{T<:AbstractFloat} end

"""
    ConstantBSL

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceBSL`](@ref).
- `z`: the BSL, considered constant in time.
- `A`: the ocean surface area, considered constant in time.
"""
mutable struct ConstantBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
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
"""
mutable struct ConstantOceanSurfaceBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    A::T
end

ConstantOceanSurfaceBSL(; ref = ReferenceBSL()) = ConstantOceanSurfaceBSL(ref, ref.z, ref.A)

"""
    PiecewiseConstantOceanSurfaceBSL{T}

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceBSL`](@ref).
- `z`: the BSL at current time step.
- `A`: the ocean surface at current time step.
"""
mutable struct PiecewiseConstantOceanSurfaceBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    A::T
end

PiecewiseConstantOceanSurfaceBSL(; ref = ReferenceBSL()) = 
    PiecewiseConstantOceanSurfaceBSL(ref, ref.z, ref.A)

mutable struct ExternalBSL{T, R<:ReferenceBSL{T}} <: AbstractBSL{T}
    ref::R
    z::T
    A::T
    t_vec::Vector{T}
    z_vec::Vector{T}
    z_itp::TimeInterpolation0D{T}
end

mutable struct CombinedBSL{T, B1<:ExternalBSL,
    B2<:Union{ConstantOceanSurfaceBSL, PiecewiseConstantOceanSurfaceBSL}} <: AbstractBSL{T}
    bsl1::B1
    bsl2::B2
end

"""
    update_bsl!(bsl::AbstractBSL, delta_V)

Update the BSL and ocean surface based on the input `delta_V` (in m^3) and
on a subtype of [`AbstractBSL`](@ref).
"""
function update_bsl!(bsl::ConstantBSL, delta_V)
    bsl.z = bsl.ref.z
    return nothing
end

function update_bsl!(bsl::ConstantOceanSurfaceBSL, delta_V)
    bsl.A = bsl.ref.A
    bsl.z += delta_V / bsl.A
    return nothing
end

function update_bsl!(bsl::PiecewiseConstantOceanSurfaceBSL, delta_V)
    bsl.z += delta_V / bsl.A
    bsl.A = bsl.ref.A_itp(bsl.z)
    return nothing
end