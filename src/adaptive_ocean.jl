"""
    AbstractOceanSurface

Abstract type for ocean surface. Available subtypes are:
- [`ConstantOceanSurface`](@ref)
- [`PiecewiseConstantOceanSurface`](@ref)
- [`PiecewiseLinearOceanSurface`](@ref), which is only available if `using NLsolve`.
"""
abstract type AbstractOceanSurface{T<:AbstractFloat} end

const A_OCEAN_PD = 3.625e14     # Ocean surface (m^2) as in Goelzer (2020, before Eq. (9))

"""
    ConstantBSL
"""
mutable struct ConstantBSL{T<:AbstractFloat}
    z::T = 0.0
end

"""
    ConstantOceanSurface{T}

A `mutable struct` containing:
- `z`: the BSL at current time step.
- `A`: the ocean surface at current time step.

The BSL can be updated by running:
```julia
os = ConstantOceanSurface{T}()
update_bsl!(os, delta_V)
```
"""
@kwdef mutable struct ConstantOceanSurface{T} <: AbstractOceanSurface{T}
    z::T = 0
    A::T = A_OCEAN_PD
end

"""
    PiecewiseConstantOceanSurface{T}

A `mutable struct` containing:
- `z`: the BSL at current time step.
- `A`: the ocean surface at current time step.
- `A_itp`: an interpolator of ocean surface over BSL.
- `A_pd`: the present-day ocean surface.

The BSL can be updated by running:
```julia
os = PiecewiseConstantOceanSurface{T}()
update_bsl!(os, delta_V)
```
"""
mutable struct PiecewiseConstantOceanSurface{T} <: AbstractOceanSurface{T}
    z::T
    A::T
    A_itp::Interpolation0D{T}
    A_pd::T
end

function PiecewiseConstantOceanSurface(; A_ocean_pd = A_OCEAN_PD, z0 = 0.0)
    z, A = load_oceansurfacefunction(verbose = false)
    A_itp = Interpolation0D(z, A)
    return PiecewiseConstantOceanSurface(z0, A_itp(z0), A_itp, A_ocean_pd)
end

"""
    PiecewiseLinearOceanSurface{T}

A `mutable struct` containing:
- `z_k`: the BSL at current time step `k`.
- `A_k`: the ocean surface at current time step `k`.
- `z`: a vector of BSL values used as knots for interpolation.
- `A`: a vector of ocean surface values used as knots for interpolation.
- `A_itp`: an interpolator of ocean surface over BSL.
- `A_pd`: the present-day ocean surface.
- `residual`: residual of the nonlinear equation solved numerically.

The BSL can be updated by running:
```julia
using NLsolve
os = PiecewiseLinearOceanSurface{T}()
update_bsl!(os, delta_V)
```

Note that, unlike [`ConstantOceanSurface`](@ref) and [`PiecewiseConstantOceanSurface`](@ref), this will only work if `using NLsolve`.
"""
mutable struct PiecewiseLinearOceanSurface{T} <: AbstractOceanSurface{T}
    z_k::T
    A_k::T
    z::Vector{T}
    A::Vector{T}
    A_itp::Interpolation0D{T}
    A_pd::T
    residual::T
end

function PiecewiseLinearOceanSurface(; A_ocean_pd = A_OCEAN_PD, z0 = 0.0)
    z, A, itp = load_oceansurfacefunction(T = T, verbose = false)
    A_itp(z) = A_ocean_pd / itp(T(0.0)) * itp(z)
    return PiecewiseLinearOceanSurface(z0, A_itp(z0), z, A, A_itp, A_ocean_pd, T(1e10))
end

"""
    update_bsl!(os::AbstractOceanSurface{T}, delta_V::T) where {T<:AbstractFloat}

Update the BSL and ocean surface based on the input `delta_V` (in m^3).
This function is defined for all subtypes of [`AbstractOceanSurface`](@ref).
"""
function update_bsl!(os::ConstantBSL{T}, delta_V::T) where {T<:AbstractFloat}
    os.z = os.z_ref
    return nothing
end

function update_bsl!(os::ConstantOceanSurface{T}, delta_V::T) where {T<:AbstractFloat}
    os.z_k += delta_V / os.A_k
    return nothing
end

function update_bsl!(os::PiecewiseConstantOceanSurface{T}, delta_V::T) where {T<:AbstractFloat}
    os.z += delta_V / os.A
    os.A = os.A_itp(os.z)
    return nothing
end

function update_bsl!(osc::OceanSurfaceChange{T}, delta_V::T) where {T<:AbstractFloat}
    if delta_V != 0
        osc.z_k += delta_V / osc.A_itp(osc.z_k)
        osc.A_k = osc.A_itp(osc.z_k)
    end
end