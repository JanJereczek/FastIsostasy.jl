# Extension name per GPU vendor. Each such extension defines a local `deviceinfo()`
# (looked up below via `Base.get_extension`) and nothing else vendor-specific —
# array allocation goes through KernelAbstractions and the FFT planner picks its
# flags off the backend, so a new vendor needs no code beyond this pair.
const GPU_EXTENSIONS = (
    :FastIsostasyCUDAExt => "CUDA.jl",
    :FastIsostasyAMDGPUExt => "AMDGPU.jl",
)

"""
    deviceinfo()

Print version information for whichever GPU package is currently loaded. Requires
one of `using CUDA` / `using AMDGPU`; errors if none is loaded.
"""
function deviceinfo()
    for (name, pkg) in GPU_EXTENSIONS
        ext = Base.get_extension(@__MODULE__, name)
        ext === nothing || return ext.deviceinfo()
    end
    error(
        "No GPU package loaded. Add one of " *
        join(("`using $(pkg)`" for (_, pkg) in GPU_EXTENSIONS), ", ") *
        " before calling `deviceinfo()`.",
    )
end

"""
    cudainfo()

!!! warning "Deprecated"
    Use [`deviceinfo`](@ref), which reports whichever GPU backend is loaded rather
    than assuming CUDA.
"""
function cudainfo()
    Base.depwarn("`cudainfo()` is deprecated, use `deviceinfo()`.", :cudainfo)
    return deviceinfo()
end

#####################################################
# Unit conversion utils
#####################################################

global SECONDS_PER_YEAR = 60^2 * 24 * 365.25

"""
$(TYPEDSIGNATURES)

Convert input time `t` from years to seconds.
"""
function years2seconds(t::T) where {T<:AbstractFloat}
    return t * T(SECONDS_PER_YEAR)
end

"""
$(TYPEDSIGNATURES)

Convert input time `t` from seconds to years.
"""
function seconds2years(t::T) where {T<:AbstractFloat}
    return t / T(SECONDS_PER_YEAR)
end

"""
$(TYPEDSIGNATURES)

Convert displacement rate `dudt` from ``m \\, s^{-1} ``to ``mm \\, \\mathrm{yr}^{-1} ``.
"""
function m_per_sec2mm_per_yr(dudt::Real)
    return dudt * 1e3 * SECONDS_PER_YEAR
end

#####################################################
# Array utils
#####################################################

not(x::Bool) = !x
# Complement for smooth (floating-point) masks in [0, 1].
not(x::AbstractFloat) = one(x) - x

Base.zeros(domain::RegionalDomain) = zeros(eltype(domain.x), domain.nx, domain.ny)

Base.fill(x::Real, sim::Simulation) = fill(x, sim.domain)
Base.fill(x, domain::RegionalDomain) = fill(eltype(domain.x)(x), domain.nx, domain.ny)

approx_in(item, collection, tol) = any(abs.(collection .- item) .< tol)

function corner_matrix(T, nx, ny)
    M = zeros(T, nx, ny)
    M[1, 1], M[nx, 1], M[1, ny], M[nx, ny] = T.([1, 1, 1, 1])
    return M
end

"""
$(TYPEDSIGNATURES)

Generate a vector of constant matrices from a vector of constants.
"""
function matrify(x::Vector{<:Real}, N::Int)
    return matrify(x, N, N)
end

function matrify(x::Vector{T}, nx::Int, ny::Int) where {T<:Real}
    X = zeros(T, nx, ny, length(x))
    @inbounds for i in eachindex(x)
        X[:, :, i] = fill(x[i], nx, ny)
    end
    return X
end

#####################################################
# Math utils
#####################################################

"""
$(TYPEDSIGNATURES)

Compute `Z = f(X,Y)` with `f` a Gaussian function parametrized by mean
`mu` and covariance `sigma`.

"""
function gauss_distr(
    X::M,
    Y::M,
    mu::Vector{T},
    sigma::Matrix{T},
) where {T<:AbstractFloat,M<:Matrix{T}}
    k = length(mu)
    G = similar(X)
    invsigma = inv(sigma)
    invsqrtdetsigma = 1/sqrt(det(sigma))
    @inbounds for i in axes(X, 1), j in axes(X, 2)
        G[i, j] =
            (2*π)^(-k/2) *
            invsqrtdetsigma *
            exp(
                -0.5 *
                ([X[i, j], Y[i, j]] .- mu)' *
                invsigma *
                ([X[i, j], Y[i, j]] .- mu),
            )
    end
    return G
end

function generate_gaussian_field(
    domain::RegionalDomain{T,M},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::Matrix{T},
) where {T<:AbstractFloat,M<:Matrix{T}}
    G = gauss_distr(domain.X, domain.Y, xy_peak, sigma)
    G = G ./ maximum(G) .* z_peak
    return fill(z_background, domain.nx, domain.ny) + G
end

#####################################################
# Quadrature utils
#####################################################

"""
$(TYPEDSIGNATURES)

Return support points and associated coefficients with specified Type
for Gauss-Legendre quadrature.
"""
function get_quad_coeffs(T::Type, n::Int)
    x, w = gausslegendre(n)
    return T.(x), T.(w)
end

"""
$(TYPEDSIGNATURES)

Compute 1D Gauss-Legendre quadrature of `f` between `x1` and `x2`
based on `n` support points.
"""
function quadrature1D(
    f::Union{Function,Interpolations.Extrapolation},
    n::Int,
    x1::T,
    x2::T,
) where {T<:AbstractFloat}
    x, w = get_quad_coeffs(T, n)
    m, p = get_normalized_lin_transform(x1, x2)
    sum = 0
    @inbounds for i = 1:n
        sum = sum + f(normalized_lin_transform(x[i], m, p)) * w[i] / m
    end
    return sum
end

"""
$(TYPEDSIGNATURES)

Return the integration of `f` over [`x1, x2`] x [`y1, y2`] with `x, w` some pre-computed
support points and coefficients of the Gauss-Legendre quadrature.
"""
function quadrature2D(
    f::Function,
    x::Vector{T},
    w::Vector{T},
    x1::T,
    x2::T,
    y1::T,
    y2::T,
) where {T<:AbstractFloat}

    n = length(x)
    mx, px = get_normalized_lin_transform(x1, x2)
    my, py = get_normalized_lin_transform(y1, y2)
    sum = T(0)
    @inbounds for i = 1:n, j = 1:n
        sum =
            sum +
            f(
                normalized_lin_transform(x[i], mx, px),
                normalized_lin_transform(x[j], my, py),
            ) *
            w[i] *
            w[j] / mx / my
    end
    return sum
end

"""
$(TYPEDSIGNATURES)

Return parameters of linear function mapping `x1, x2` onto `-1, 1`.
"""
function get_normalized_lin_transform(x1::T, x2::T) where {T<:AbstractFloat}
    x1_norm, x2_norm = T(-1), T(1)
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

"""
$(TYPEDSIGNATURES)

Apply normalized linear transformation with slope `m` and bias `p` on `y`.
"""
function normalized_lin_transform(y::T, m::T, p::T) where {T<:AbstractFloat}
    return (y-p)/m
end

#####################################################
# Kernel utils
#####################################################

"""
$(TYPEDSIGNATURES)

Allocate a zeroed array on a backend. With a `domain` it returns an `nx × ny`
array of the domain's element type on the domain's backend; the explicit form
takes any element type and shape.
"""
kernelzeros(domain::RegionalDomain) =
    kernelzeros(domain.backend, eltype(domain), domain.nx, domain.ny)
kernelzeros(backend::Backend, T, dims::Integer...) =
    KernelAbstractions.zeros(backend, T, dims...)

"""
$(TYPEDSIGNATURES)

True when `domain`'s arrays live in host memory.
"""
on_host(domain::RegionalDomain) = domain.backend isa CPU

# NOTE: `kernelcollect` and the `on_host` branch in `init_problem!` are not really
# about hardware — they exist because a CPU mask comparison yields a `BitArray`,
# which the state structs want materialised as a dense `Array{Bool}`, while a
# device array is already dense and must not be pulled back to the host. Forcing
# dense Bool at the point the masks are *built* would remove both. Left as-is here
# to keep this refactor behaviour-preserving.
kernelcollect(X, domain) = on_host(domain) ? collect(X) : X

"""
$(TYPEDSIGNATURES)

Move `X` onto `backend`, leaving it untouched if it is already there.

Allocation goes through `KernelAbstractions.allocate`, so this works for every
KA backend without FastIsostasy ever naming a vendor array type.
"""
function kernelpromote(X::AbstractArray, backend::Backend)
    _lives_on(X, backend) && return X
    src = _dense_host(X)
    Y = KernelAbstractions.allocate(backend, eltype(src), size(src)...)
    copyto!(Y, src)
    return Y
end

# A `BitArray` — which is what every mask comparison returns on the host — packs 64
# booleans per word, and no GPU backend can `copyto!` from that layout. Materialise
# it as a dense `Array{Bool}` first. (The old code got this for free because
# `CuArray(::BitArray)` has a converting constructor; `copyto!` does not.)
_dense_host(X::AbstractArray) = X
_dense_host(X::BitArray) = Array(X)
kernelpromote(X::Vector{<:AbstractArray}, backend::Backend) =
    [kernelpromote(x, backend) for x in X]

# Only a dense `Array` counts as "already on the CPU": a `BitArray` (what a mask
# comparison returns) is deliberately re-materialised as `Array{Bool}`, which is
# what the pre-backend `isa(X, Array)` test did too.
#
# `KernelAbstractions.get_backend` *throws* for host array types it does not know —
# `BitArray` among them — so both host cases are settled before asking it. `backend`
# is a singleton and `X`'s type is known at the call site, so all of this folds away.
function _lives_on(X::AbstractArray, backend::Backend)
    X isa Array && return backend isa CPU
    X isa BitArray && return false
    return get_backend(X) === backend
end


# function remake!(sim::Simulation)

#     T = Float64
#     (; domain, ref, now) = sim

#     now.u .= ref.u
#     now.dudt .= T.(0.0)
#     now.ue .= ref.ue
#     now.u_eq .= ref.u
#     now.ucorner = T(0.0)
#     now.H_ice .= ref.H_ice
#     now.H_water .= ref.H_water
#     now.columnanoms = ColumnAnomalies(domain)
#     now.z_b .= ref.z_b
#     now.bsl = ref.bsl
#     now.dz_ss .= T.(0.0)
#     now.z_ss .= ref.z_ss
#     now.V_af = ref.V_af
#     now.V_pov = ref.V_pov
#     now.V_den = ref.V_den
#     now.maskgrounded .= ref.maskgrounded
#     now.maskocean .= ref.maskocean
#     now.osc = OceanSurfaceChange(T = T, z0 = ref.bsl)
#     now.count_sparse_updates = 0
#     now.k = 1

#     return nothing
# end

#####################################################
# Example utils
#####################################################

function mask_disc(X, Y, R; center = [0, 0])
    return mask_disc(sqrt.((X .- center[1]) .^ 2 + (Y .- center[2]) .^ 2), R)
end

function mask_disc(r::AbstractMatrix{T}, R) where {T<:AbstractFloat}
    return T.(r .< R)
end

function uniform_ice_cylinder(
    domain::RegionalDomain,
    R::T,
    H::T;
    center::Vector{T} = T.([0.0, 0.0]),
) where {T<:AbstractFloat}
    M = mask_disc(domain.X, domain.Y, R, center = center)
    return T.(M .* H)
end

function stereo_ice_cylinder(domain::RegionalDomain, R, H)
    M = mask_disc(domain.R, R)
    return M .* H
end

function stereo_ice_cap(
    domain::RegionalDomain,
    alpha_deg::T,
    H::T,
) where {T<:AbstractFloat}
    alpha = deg2rad(alpha_deg)
    M = domain.Theta .< alpha
    return H .* sqrt.(M .* (cos.(domain.Theta) .- cos(alpha)) ./ (1 - cos(alpha)))
end
