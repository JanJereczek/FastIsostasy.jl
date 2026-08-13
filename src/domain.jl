"""
$(TYPEDSIGNATURES)

Abstract type for domain representation in the model.

# Available subtypes:
- [`RegionalDomain`](@ref)
- [`GlobalDomain`](@ref): Not implemented yet, but hopefully in v3.0!
"""
abstract type AbstractDomain end

"""
$(TYPEDSIGNATURES)

Not implemented yet!

Version 3.0 will allow for global domains.
"""
struct GlobalDomain <: AbstractDomain end

"""
$(TYPEDSIGNATURES)

Define a regional domain, including its geometry and the architecture it should be running on.

# Initialization

For a square domain with half-width `W` and `2^n` grid points in each dimension:
```julia
domain = RegionalDomain(W, n)
```

For a rectangular domain with half-widths `Wx` and `Wy` and `nx` and `ny` grid points in each dimension:
```julia
domain = RegionalDomain(Wx, Wy, nx, ny)
```

For a rectangular domain with spanning vectors `x` and `y`:
```julia
domain = RegionalDomain(x, y)           # rectangular domain: spanning vectors x, y
```

# Hardware

The `backend` keyword selects where the arrays live and where the kernels run. It
takes a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend, so FastIsostasy is not tied to any single GPU vendor:

| `backend` | needs |
|---|---|
| `CPU()` (default) | — |
| `CUDABackend()` | `using CUDA` |
| `ROCBackend()` | `using AMDGPU` |
| `MetalBackend()` | `using Metal` |
| `oneAPIBackend()` | `using oneAPI` |

```julia
using CUDA
domain = RegionalDomain(3000e3, 7, backend = CUDABackend())
```

Everything downstream — `SolidEarth`, `BoundaryConditions`, `Simulation` — picks
the backend up from the domain, so this is the only place hardware is named.

!!! compat "Deprecated: `arraykernel`"
    Before v2.1 the hardware was chosen with an array constructor
    (`arraykernel = CuArray`). That keyword still works but warns; pass `backend`
    instead.
"""
struct RegionalDomain{T,L,M,B} <: AbstractDomain

    Wx::T                       # Domain half-width in x (m)
    Wy::T                       # Domain half-width in y (m)
    nx::Int                     # Number of grid points in x-dimension
    ny::Int                     # Number of grid points in y-dimension
    mx::Int                     # nx/2
    my::Int                     # ny/2
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
    Dx::M                       # dx matrix accounting for distortion.  TODO: macro
    Dy::M                       # dy matrix accounting for distortion.  TODO: macro
    A::M                        # area (accounting for distortion).     TODO: macro
    correct_distortion::Bool
    zeros::M                     # a zero matrix of size nx x ny
    pseudodiff::M               # pseudodiff operator as matrix (Hadamard product)
    # Which hardware the arrays live on, as a `KernelAbstractions.Backend`:
    # `CPU()`, `CUDABackend()`, `ROCBackend()`, `MetalBackend()`, `oneAPIBackend()`.
    # FastIsostasy never names a vendor array type — allocation goes through
    # `kernelzeros`/`kernelpromote`, which call `KernelAbstractions.allocate`.
    #
    # KA backends are singletons, so storing the *instance* still lifts the choice
    # into the type domain: `B` is a compile-time constant and every
    # `backend`-dependent branch folds away, exactly as the old `Type{K}` field did.
    # `B` is the *last* parameter so that partially applied signatures
    # (`RegionalDomain{T, L, M}`) keep dispatching.
    backend::B
end

function RegionalDomain(W::T, n::Int; kwargs...) where {T<:AbstractFloat}
    Wx, Wy = W, W
    nx, ny = 2^n, 2^n
    return RegionalDomain(Wx, Wy, nx, ny; kwargs...)
end

function RegionalDomain(
    Wx::T,
    Wy::T,
    nx::Int,
    ny::Int;
    kwargs...,
) where {T<:AbstractFloat}
    mx, my = nx ÷ 2, ny ÷ 2
    dx = 2*Wx / nx
    dy = 2*Wy / ny
    x = collect(range(-Wx+dx, stop = Wx, length = nx))
    y = collect(range(-Wy+dy, stop = Wy, length = ny))
    return RegionalDomain(x, y, dx, dy, Wx, Wy, nx, ny, mx, my; kwargs...)
end

function RegionalDomain(x::Vector{T}, y::Vector{T}; kwargs...) where {T<:AbstractFloat}
    nx = length(x)
    ny = length(y)
    mx, my = nx ÷ 2, ny ÷ 2

    centering_tolerance = 1e3
    if mean(x) > centering_tolerance || mean(y) > centering_tolerance
        error("x and y must be centered around zero.")
    end
    Wx, Wy = maximum(abs.(x)), maximum(abs.(y))

    if std(diff(x)) .> 1e-5 || std(diff(y)) .> 1e-5
        error("x and y must be regularly spaced.")
    end

    dx = mean(diff(x))
    dy = mean(diff(y))

    return RegionalDomain(x, y, dx, dy, Wx, Wy, nx, ny, mx, my; kwargs...)
end

function RegionalDomain(
    x::Vector{T},
    y::Vector{T},
    dx::T,
    dy::T,
    Wx::T,
    Wy::T,
    nx::Int,
    ny::Int,
    mx::Int,
    my::Int;
    backend = CPU(),
    arraykernel = nothing,      # deprecated, see `_backend_from_arraykernel`
    lat_ref::T = T(-71.0),      # Reference latitude for scale factor
    lon_ref::T = T(0.0),        # Reference longitude for scale factor
    lat_0::T = T(-90.0),        # Latitude of center point (allows oblique proj)
    lon_0::T = T(0.0),          # Longitude of center point (allows oblique proj)
    proj_lonlat = "EPSG:4326",
    proj_target = "+proj=stere +datum=WGS84",
    correct_distortion::Bool = false,
) where {T<:AbstractFloat}

    X, Y = meshgrid(x, y)
    zeros = fill(T(0), nx, ny)
    R = get_r.(X, Y)

    lonlat2target = Proj.Transformation(
        proj_lonlat,
        "$proj_target +lat_0=$lat_0 +lat_ts=$lat_ref +lon_0=$lon_0 +lon_ts=$lon_ref",
        always_xy = true,
    )
    target2lonlat = Proj.inv(lonlat2target)
    coords = target2lonlat.(X, Y)
    Lon = T.(map(x -> x[1], coords))
    Lat = T.(map(x -> x[2], coords))

    if correct_distortion
        K = T.(scalefactor(Lat, lat_ref))
        if approx_in(0.0, x, 1e3) || approx_in(0.0, y, 1e3)
            K[mx, my] = mean([K[mx-1, my], K[mx+1, my], K[mx, my-1], K[mx, my+1]])
        end
    else
        K = fill(T(1), nx, ny)
    end
    Theta = dist2angulardist.(K .* R)

    # Differential operators in Fourier space
    pseudodiff, _, _ = get_differential_fourier(Wx, Wy, nx, ny)

    # Avoid division by zero in scaled_pseudodiff_inv (src/solidearth.jl) by
    # setting the DC mode to the mean of its neighbours rather than exactly
    # zero. Do NOT scale this down further (e.g. by 1e-3): under a normalised
    # BC (CornerBC/BorderBC), apply_bc! projects out the constant mode every
    # RHS evaluation, so this value never reaches the solution there — scaling
    # it down only manufactures a large constant that is created and cancelled
    # every step (costing precision, esp. in Float32) and, under NoBC, turns
    # the mean mode into an artificial stiff eigenmode that throttles explicit
    # time steppers for no dynamical reason. (An alternative, exact-removal fix
    # is scaled_pseudodiff_inv[1,1] = 0 in src/solidearth.jl, per Bueler et al.
    # 2007's corner normalisation — the constant is then supplied by the BC,
    # not the dynamics; it only differs from this under NoBC.) See
    # fastisostasy-roadmap/stabilise_dt.md §1/§4.
    pseudodiff[1, 1] = mean([pseudodiff[1, 2], pseudodiff[2, 1]])

    backend = _backend_from_arraykernel(backend, arraykernel)
    zeros, K, pseudodiff = kernelpromote([zeros, K, pseudodiff], backend)

    i1, i2 = samesize_conv_indices(nx, mx)
    j1, j2 = samesize_conv_indices(ny, my)
    convo_offset = (ny - nx) ÷ 2
    convo_offset = 0

    return RegionalDomain(
        Wx,
        Wy,
        nx,
        ny,
        mx,
        my,
        dx,
        dy,
        x,
        y,
        X,
        Y,
        i1,
        i2,
        j1,
        j2,
        convo_offset,
        R,
        Theta,
        Lat,
        Lon,
        K,
        K .* dx,
        K .* dy,
        (dx * dy) .* K .^ 2,
        correct_distortion,
        zeros,
        pseudodiff,
        backend,
    )
end

"""
$(TYPEDSIGNATURES)

Resolve the hardware choice, accepting the deprecated `arraykernel` keyword.

Before v2.1 the hardware was selected by passing an array *constructor*
(`arraykernel = CuArray`); it is now a `KernelAbstractions.Backend`
(`backend = CUDABackend()`), which is vendor-neutral and is the same object the
`@kernel` launches already dispatch on. `arraykernel` still works but warns.
"""
function _backend_from_arraykernel(backend, arraykernel)
    arraykernel === nothing && return backend
    Base.depwarn(
        "`RegionalDomain(...; arraykernel = $arraykernel)` is deprecated. Pass a " *
        "KernelAbstractions backend instead: `backend = CPU()`, " *
        "`backend = CUDABackend()` (CUDA.jl), `backend = ROCBackend()` (AMDGPU.jl), " *
        "`backend = MetalBackend()` (Metal.jl) or `backend = oneAPIBackend()` " *
        "(oneAPI.jl).",
        :RegionalDomain,
    )
    arraykernel === Array && return CPU()
    # A vendor array type: ask KernelAbstractions which backend owns it, using a
    # 0-element instance so nothing meaningful is allocated on the device.
    return get_backend(arraykernel(undef, ntuple(_ -> 0, 2)...))
end

Base.eltype(domain::RegionalDomain) = eltype(domain.x)

function Base.show(io::IO, ::MIME"text/plain", domain::RegionalDomain)
    descriptors = [
        "nx, ny" => [domain.nx, domain.ny],
        "dx, dy" => [domain.dx, domain.dy],
        "Wx, Wy" => [domain.Wx, domain.Wy],
        "eltype" => eltype(domain),
        "backend" => domain.backend,
        "correct_distortion" => domain.correct_distortion,
    ]
    padlen = maximum(length(d[1]) for d in descriptors) + 2
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end