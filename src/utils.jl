cudainfo() = CUDA.versioninfo()

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
function gauss_distr(X::M, Y::M, mu::Vector{T}, sigma::Matrix{T}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    k = length(mu)
    G = similar(X)
    invsigma = inv(sigma)
    invsqrtdetsigma = 1/sqrt(det(sigma))
    @inbounds for i in axes(X,1), j in axes(X,2)
        G[i, j] = (2*π)^(-k/2) * invsqrtdetsigma * exp( 
            -0.5 * ([X[i,j], Y[i,j]] .- mu)' * invsigma * ([X[i,j], Y[i,j]] .- mu) )
    end
    return G
end

function generate_gaussian_field(
    domain::RegionalDomain{T, M},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::Matrix{T},
) where {T<:AbstractFloat, M<:Matrix{T}}
    if domain.nx == domain.ny
        N = domain.nx
    else
        error("Automated generation of Gaussian parameter fields only supported for" *
            "square domains.")
    end
    G = gauss_distr( domain.X, domain.Y, xy_peak, sigma )
    G = G ./ maximum(G) .* z_peak
    return fill(z_background, N, N) + G
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
function quadrature1D(f::Union{Function, Interpolations.Extrapolation},
    n::Int, x1::T, x2::T) where {T<:AbstractFloat}
    x, w = get_quad_coeffs(T, n)
    m, p = get_normalized_lin_transform(x1, x2)
    sum = 0
    @inbounds for i=1:n
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
    x1::T, x2::T,
    y1::T, y2::T,
) where {T<:AbstractFloat}

    n = length(x)
    mx, px = get_normalized_lin_transform(x1, x2)
    my, py = get_normalized_lin_transform(y1, y2)
    sum = T(0)
    @inbounds for i=1:n, j=1:n
        sum = sum + f(
            normalized_lin_transform(x[i], mx, px),
            normalized_lin_transform(x[j], my, py),
        ) * w[i] * w[j] / mx / my
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

kernelzeros(domain) = domain.arraykernel(zeros(domain))

function kernelcollect(X, domain)
    if not(domain.use_cuda)
        return collect(X)
    else
        return X
    end
end

"""
$(TYPEDSIGNATURES)

Promote X to the kernel (`Array` or `CuArray`) specified by `arraykernel`.
"""
function kernelpromote(X, arraykernel)
    if isa(X, arraykernel)
        return X
    else
        return arraykernel(X)
    end
end
kernelpromote(X::Vector, arraykernel) = [arraykernel(x) for x in X]


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

function mask_disc(r::KernelMatrix{T}, R) where {T<:AbstractFloat}
    return T.(r .< R)
end

function uniform_ice_cylinder(domain::RegionalDomain, R::T, H::T;
    center::Vector{T} = T.([0.0, 0.0])) where {T<:AbstractFloat}
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
    return H .* sqrt.( M .* (cos.(domain.Theta) .- cos(alpha)) ./ (1 - cos(alpha)) )
end