"""
    get_r(x::T, y::T) where {T<:Real}

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:Real} = sqrt(x^2 + y^2)

"""
    meshgrid(x, y)

Return a 2D meshgrid spanned by `x, y`.
"""
function meshgrid(x::V, y::V) where {T<:AbstractFloat, V<:AbstractVector{T}}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return x * one_y', one_x * y'
end

"""
    dist2angulardist(r::Real)

Convert Euclidean to angular distance along great circle.
"""
function dist2angulardist(r::T) where {T<:AbstractFloat}
    R = T(6371e3)       # radius at equator
    return 2 * atan( r / (2 * R) )
end

"""
    lon360tolon180(lon, X)

Convert longitude and field from `lon=0:360` to `lon=-180:180`.
"""
function lon360tolon180(lon, X)
    permidx = lon .> 180
    lon180 = vcat(lon[permidx] .- 360, lon[not.(permidx)])
    X180 = cat(X[permidx, :, :], X[not.(permidx), :, :], dims=1)
    return lon180, X180
end

"""
    XY2LonLat(X, Y, proj)

Convert Cartesian coordinates `(X, Y)` to longitude-latitude `(Lon, Lat)`
using the projection `proj`.
"""
function XY2LonLat(X, Y, proj)
    coords = proj.(X, Y)
    Lon = map(x -> x[1], coords)
    Lat = map(x -> x[2], coords)
    return Lon, Lat
end

"""
    scalefactor(Lat, lat_s; kwargs...)

Compute scaling factor of Lambert conformal conic projection for a given `Lat::AbstractMatrix`.
This follows the example *Polar aspect with know scale factor*, beginning on page 314 of
[snyder-projections-1987](@citet).
"""
function scalefactor(
    Lat::M,     # latitude array
    lat_s::T;   # standard parallel
    kwargs...,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    lat_s = deg2rad(lat_s)
    t_s = lambert_t(lat_s)
    m_s = lambert_m(lat_s)
    r_equator = 6371e3

    return lambert_k.(
        lambert_rho.(r_equator, m_s, lambert_t.(deg2rad.(Lat); kwargs...), t_s),
        r_equator,
        lambert_m.(deg2rad.(Lat); kwargs...),
    )
end

"""
    lambert_t(phi; e = 0.0819919)

Compute `t` following (Eq. 15-9) of [snyder-projections-1987](@citet) for a given latitude
`phi` and eccentricity `e`.
"""
lambert_t(phi; e = 0.0819919) = tan(π/4+phi/2) / ((1+e*sin(phi))/(1 - e * sin(phi)))^(e/2)

"""
    lambert_m(phi; e = 0.0819919)

Compute `m` following (Eq. 14-15) of [snyder-projections-1987](@citet) for a given latitude
`phi` and eccentricity `e`.
"""
lambert_m(phi; e = 0.0819919) = cos(phi) / sqrt(1 - e^2 * sin(phi)^2)

"""
    lambert_rho(r, mc, t, tc)

Compute `rho` following (Eq. 21-34) of [snyder-projections-1987](@citet) for given
`r`, `mc`, and `t` and `tc`.
"""
lambert_rho(r, mc, t, tc) = r .* mc .* t ./ tc

"""
    lambert_k(rho, r, m)

Compute `k` following (Eq. 21-32) of [snyder-projections-1987](@citet) for given
`rho`, `r`, and `m`.
"""
lambert_k(rho, r, m) = rho ./ r ./ m


"""
    scalefactor(lat::T, lon::T, lat_0::T, lon_0::T) where {T<:Real}
    scalefactor(lat::M, lon::M, lat_0::T, lon_0::T) where {T<:Real, M<:KernelMatrix{T}}

Compute scaling factor of stereographic projection for a given `(lat, lon)` and origin
`(lat_0, lon_0)`. Angles must be provided in radians. Reference:
[snyder-projections-1987](@citet), p. 157, eq. (21-4).

Note: this version of the function does not support a standard parallel different from
the pole.
"""
function scalefactor(lat::T, lon::T, lat_0::T, lon_0::T; k0::T = T(1)) where {T<:Real}
    return 2*k0 / (1 + sin(lat_0)*sin(lat) + cos(lat_0)*cos(lat)*cos(lon-lon_0))
end

function scalefactor(lat::M, lon::M, lat_0::T, lon_0::T; kwargs...,
    ) where {T<:Real, M<:KernelMatrix{T}}
    K = similar(lat)
    @inbounds for idx in CartesianIndices(lat)
        K[idx] = scalefactor(lat[idx], lon[idx], lat_0, lon_0; kwargs...)
    end
    return K
end