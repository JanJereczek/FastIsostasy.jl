const A_OCEAN_PD = 3.625e14     # Ocean surface (m^2) as in Goelzer (2020, before Eq. (9))

"""
    ReferenceOcean{T<:AbstractFloat, I<:Interpolations.Extrapolation}
    ReferenceOcean(z = 0; T = Float32, itp_kwargs = (extrapolation_bc = Flat()))

A `struct` used in all subtypes of `AbstractOceanSurface` to define a fixed
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
ref = ReferenceOcean()              # assume BSL = 0
```

Custom:
```julia
ref = ReferenceOcean(z = 0.1)       # assume BSL = 0.1 m and compute A accordingly
```
"""
struct ReferenceOcean{T<:AbstractFloat, I<:Interpolations.Extrapolation}
    z::T
    A::T
    z_vec::Vector{T}
    A_vec::Vector{T}
    A_itp::I
end

function ReferenceOcean(z; T = Float32, itp_kwargs = (extrapolation_bc = Flat()))
    z_vec, A_vec = load_oceansurface_data(T = T, verbose = false)
    itp = linear_interpolation(z_vec, A_vec; itp_kwargs...)
    A_itp(zz) = T(A_OCEAN_PD) / itp(T(0)) * itp(zz)
    A = A_itp(z)
    return ReferenceOcean(T(z), T(A), z_vec, A_vec, A_itp)
end

ReferenceOcean() = ReferenceOcean(0)

Base.eltype(ref::ReferenceOcean{T}) where {T<:AbstractFloat} = T

"""
    AbstractOceanSurface

Abstract type for ocean surface. Available subtypes are:
- [`ConstantBSL`](@ref)
- [`ConstantOceanSurface`](@ref)
- [`PiecewiseConstantOceanSurface`](@ref)
- [`PiecewiseLinearOceanSurface`](@ref)

All subtypes implement the `update_bsl!` function:
```julia
T, delta_V = Float64, 1.0e9                 # Example values
os = PiecewiseConstantOceanSurface{T}()     # or any other subtype!
update_bsl!(os, delta_V)
```
"""
abstract type AbstractOceanSurface{T<:AbstractFloat} end

"""
    ConstantSeaLevel

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceOcean`](@ref).
- `z`: the BSL, considered constant in time.
- `A`: the ocean surface area, considered constant in time.
"""
mutable struct ConstantSeaLevel{T, R<:ReferenceOcean{T}} <: AbstractOceanSurface{T}
    ref::R
    z::T
    A::T
end

ConstantSeaLevel(; ref = ReferenceOcean()) = ConstantSeaLevel(ref, ref.z, ref.A)

"""
    ConstantOceanSurface{T}

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceOcean`](@ref).
- `z`: the BSL at current time step.
- `A`: the ocean surface area, considered constant in time.
"""
mutable struct ConstantOceanSurface{T, R<:ReferenceOcean{T}} <: AbstractOceanSurface{T}
    ref::R
    z::T
    A::T
end

ConstantOceanSurface(; ref = ReferenceOcean()) = ConstantOceanSurface(ref, ref.z, ref.A)

"""
    PiecewiseConstantOceanSurface{T}

A `mutable struct` containing:
- `ref`: an instance of [`ReferenceOcean`](@ref).
- `z`: the BSL at current time step.
- `A`: the ocean surface at current time step.
"""
mutable struct PiecewiseConstantOceanSurface{T, R<:ReferenceOcean{T}} <: AbstractOceanSurface{T}
    ref::R
    z::T
    A::T
end

PiecewiseConstantOceanSurface(; ref = ReferenceOcean()) = 
    PiecewiseConstantOceanSurface(ref, ref.z, ref.A)

"""
    update_bsl!(os::AbstractOceanSurface, delta_V)

Update the BSL and ocean surface based on the input `delta_V` (in m^3) and
on a subtype of [`AbstractOceanSurface`](@ref).
"""
function update_bsl!(os::ConstantSeaLevel, delta_V)
    os.z = os.ref.z
    return nothing
end

function update_bsl!(os::ConstantOceanSurface, delta_V)
    os.A = os.ref.A
    os.z += delta_V / os.A
    return nothing
end

function update_bsl!(os::PiecewiseConstantOceanSurface, delta_V)
    os.z += delta_V / os.A
    os.A = os.ref.A_itp(os.z)
    return nothing
end