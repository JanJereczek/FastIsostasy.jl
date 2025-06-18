#########################################################
# Computation domain
#########################################################
"""
    ComputationDomain
    ComputationDomain(W, n)
    ComputationDomain(Wx, Wy, nx, ny)

Return a struct containing all information related to geometry of the domain
and potentially used parallelism. To initialize one with `2*W` and `2^n` grid cells:

```julia
Omega = ComputationDomain(W, n)
```

If a rectangular domain is needed, run:

```julia
Omega = ComputationDomain(Wx, Wy, nx, ny)
```
"""
struct ComputationDomain{T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
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
    null::M                     # a zero matrix of size nx x ny
    pseudodiff::M               # pseudodiff operator as matrix (Hadamard product)
    # pseudodiff_inv::M
    use_cuda::Bool
    arraykernel::Any            # Array or CuArray depending on chosen hardware
end

function ComputationDomain(W::T, n::Int; kwargs...) where {T<:AbstractFloat}
    Wx, Wy = W, W
    nx, ny = 2^n, 2^n
    return ComputationDomain(Wx, Wy, nx, ny; kwargs...)
end

function ComputationDomain(Wx::T, Wy::T, nx::Int, ny::Int; kwargs...) where {T<:AbstractFloat}
    mx, my = nx ÷ 2, ny ÷ 2
    dx = 2*Wx / nx
    dy = 2*Wy / ny
    # x = collect(-Wx+dx:dx:Wx)
    # y = collect(-Wy+dy:dy:Wy)
    x = collect(range(-Wx+dx, stop = Wx, length = nx))
    y = collect(range(-Wy+dy, stop = Wy, length = ny))
    return ComputationDomain(x, y, dx, dy, Wx, Wy, nx, ny, mx, my; kwargs...)
end


function ComputationDomain(x::Vector{T}, y::Vector{T}; kwargs...) where {T<:AbstractFloat}
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

    return ComputationDomain(x, y, dx, dy, Wx, Wy, nx, ny, mx, my; kwargs...)
end

function ComputationDomain(
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
    use_cuda::Bool = false,
    lat_ref::T = T(-71.0),      # Reference latitude for scale factor
    lon_ref::T = T(0.0),        # Reference longitude for scale factor
    lat_0::T = T(-90.0),        # Latitude of center point (allows oblique proj)
    lon_0::T = T(0.0),          # Longitude of center point (allows oblique proj)
    proj_lonlat = "EPSG:4326",
    proj_target = "+proj=stere +datum=WGS84",
    correct_distortion::Bool = true,
) where {T<:AbstractFloat}

    X, Y = meshgrid(x, y)
    null = fill(T(0), nx, ny)
    R = get_r.(X, Y)

    lonlat2target = Proj.Transformation(proj_lonlat,
        "$proj_target +lat_0=$lat_0 +lat_ts=$lat_ref +lon_0=$lon_0 +lon_ts=$lon_ref",
        always_xy=true)
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

    # Avoid division by zero. Tolerance ϵ of the order of the neighboring terms.
    # Tests show that it does not lead to errors wrt analytical or benchmark solutions.
    pseudodiff[1, 1] = 1e-3 * mean([pseudodiff[1,2], pseudodiff[2,1]])
    
    arraykernel = use_cuda ? CuArray : Array
    null, K, pseudodiff = kernelpromote([null, K, pseudodiff], arraykernel)

    i1, i2 = samesize_conv_indices(nx, mx)
    j1, j2 = samesize_conv_indices(ny, my)
    convo_offset = (ny - nx) ÷ 2

    return ComputationDomain(Wx, Wy, nx, ny, mx, my, dx, dy, x, y, X, Y, i1, i2, j1, j2, convo_offset,
        R, Theta, Lat, Lon, K, K .* dx, K .* dy, (dx * dy) .* K .^ 2, correct_distortion,
        null, pseudodiff, use_cuda, arraykernel)
end