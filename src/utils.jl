
cudainfo() = CUDA.versioninfo()

#####################################################
# Unit conversion utils
#####################################################

global SECONDS_PER_YEAR = 60^2 * 24 * 365.25

"""
    years2seconds(t::Real)

Convert input time `t` from years to seconds.
"""
function years2seconds(t::T) where {T<:AbstractFloat}
    return t * T(SECONDS_PER_YEAR)
end

"""
    seconds2years(t::Real)

Convert input time `t` from seconds to years.
"""
function seconds2years(t::T) where {T<:AbstractFloat}
    return t / T(SECONDS_PER_YEAR)
end

"""
    m_per_sec2mm_per_yr(dudt::Real)

Convert displacement rate `dudt` from ``m \\, s^{-1} ``to ``mm \\, \\mathrm{yr}^{-1} ``.
"""
function m_per_sec2mm_per_yr(dudt::Real)
    return dudt * 1e3 * SECONDS_PER_YEAR
end

#####################################################
# Array utils
#####################################################

not(x::Bool) = !x
Base.fill(x::Real, fip::FastIsoProblem) = fill(x, fip.Omega)
Base.fill(x::Real, Omega::ComputationDomain) = Omega.arraykernel(fill(x, Omega.Nx, Omega.Ny))

approx_in(item, collection, tol) = any(abs.(collection .- item) .< tol)

function corner_matrix(T, Nx, Ny)
    M = zeros(T, Nx, Ny)
    M[1, 1], M[Nx, 1], M[1, Ny], M[Nx, Ny] = T.([1, 1, 1, 1])
    return M
end

"""
    matrify(x, Nx, Ny)

Generate a vector of constant matrices from a vector of constants.
"""
function matrify(x::Vector{<:Real}, N::Int)
    return matrify(x, N, N)
end

function matrify(x::Vector{T}, Nx::Int, Ny::Int) where {T<:Real}
    # X = Array{T, 3}(undef, Nx, Ny, length(x))
    X = zeros(T, Nx, Ny, length(x))
    @inbounds for i in eachindex(x)
        X[:, :, i] = fill(x[i], Nx, Ny)
    end
    return X
end

#####################################################
# Math utils
#####################################################

"""
    gauss_distr(X::KernelMatrix{T}, Y::KernelMatrix{T},
        mu::Vector{<:Real}, sigma::Matrix{<:Real})

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
    Omega::ComputationDomain{T, M},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::Matrix{T},
) where {T<:AbstractFloat, M<:Matrix{T}}
    if Omega.Nx == Omega.Ny
        N = Omega.Nx
    else
        error("Automated generation of Gaussian parameter fields only supported for" *
            "square domains.")
    end
    G = gauss_distr( Omega.X, Omega.Y, xy_peak, sigma )
    G = G ./ maximum(G) .* z_peak
    return fill(z_background, N, N) + G
end

#####################################################
# Quadrature utils
#####################################################

"""
    get_quad_coeffs(T, n)

Return support points and associated coefficients with specified Type
for Gauss-Legendre quadrature.
"""
function get_quad_coeffs(T::Type, n::Int)
    x, w = gausslegendre(n)
    return T.(x), T.(w)
end


"""
    quadrature1D(f, n, x1, x2)

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
    quadrature2D(f, x, w, x1, x2, y1, y2)

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
    get_normalized_lin_transform(x1, x2)

Return parameters of linear function mapping `x1, x2` onto `-1, 1`.
"""
function get_normalized_lin_transform(x1::T, x2::T) where {T<:AbstractFloat}
    x1_norm, x2_norm = T(-1), T(1)
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

"""
    normalized_lin_transform(y, m, p)

Apply normalized linear transformation with slope `m` and bias `p` on `y`.
"""
function normalized_lin_transform(y::T, m::T, p::T) where {T<:AbstractFloat}
    return (y-p)/m
end

#####################################################
# Kernel utils
#####################################################

function null(Omega::ComputationDomain{T, L, M}) where {T, L, M}
    return zeros(T, Omega.Nx, Omega.Ny)
end

kernelnull(Omega) = Omega.arraykernel(null(Omega))

function kernelcollect(X, Omega)
    if not(Omega.use_cuda)
        return collect(X)
    else
        return X
    end
end

"""
    kernelpromote(X, arraykernel)

Promote X to the kernel (`Array` or `CuArray`) specified by `arraykernel`.
"""
function kernelpromote(X::M, arraykernel) where {M<:AbstractArray{T}} where {T<:Real}
    if isa(X, arraykernel)
        return X
    else
        return arraykernel(X)
    end
end

kernelpromote(X::Vector, arraykernel) = [arraykernel(x) for x in X]


"""
    reinit_structs_cpu(Omega, p)

Reinitialize `Omega::ComputationDomain` and `p::LayeredEarth` on the CPU, mostly
for post-processing purposes.
"""
function reinit_structs_cpu(Omega::ComputationDomain{T, M}, p::LayeredEarth{T, M}
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    Omega_cpu = ComputationDomain(Omega.Wx, Omega.Wy, Omega.Nx, Omega.Ny, use_cuda = false)
    p_cpu = LayeredEarth(
        Omega_cpu;
        layer_boundaries = Array(p.layer_boundaries),
        layer_viscosities = Array(p.layer_viscosities),
    )
    return Omega_cpu, p_cpu
end

function choose_fft_plans(X, use_cuda)
    if use_cuda
        pfft! = CUFFT.plan_fft!(complex.(X))
        pifft! = CUFFT.plan_ifft!(complex.(X))
    else
        pfft! = plan_fft!(complex.(X))
        pifft! = plan_ifft!(complex.(X))
    end
    return pfft!, pifft!
end

function remake!(fip::FastIsoProblem)

    T = Float64
    (; Omega, ref, now) = fip

    now.u .= ref.u
    now.dudt .= T.(0.0)
    now.ue .= ref.ue
    now.u_eq .= ref.u
    now.ucorner = T(0.0)
    now.H_ice .= ref.H_ice
    now.H_water .= ref.H_water
    now.columnanoms = ColumnAnomalies(Omega)
    now.b .= ref.b
    now.bsl = ref.bsl
    now.dz_ss .= T.(0.0)
    now.z_ss .= ref.z_ss
    now.V_af = ref.V_af
    now.V_pov = ref.V_pov
    now.V_den = ref.V_den
    now.maskgrounded .= ref.maskgrounded
    now.maskocean .= ref.maskocean
    now.osc = OceanSurfaceChange(T = T, z0 = ref.bsl)
    now.countupdates = 0
    now.k = 1

    return nothing
end

#####################################################
# BC utils
#####################################################

function periodic_extension(M::Matrix{T}, Nx::Int, Ny::Int) where {T<:AbstractFloat}
    M_periodic = zeropad_extension(M, Nx, Ny)
    M_periodic[1, 2:end-1] .= M[end, :]
    M_periodic[end, 2:end-1] .= M[1, :]
    M_periodic[2:end-1, 1] .= M[:, end]
    M_periodic[2:end-1, end] .= M[:, 1]
    return M_periodic
end

function zeropad_extension(M::Matrix{T}, Nx::Int, Ny::Int) where {T<:AbstractFloat}
    M_zeropadded = fill(T(0), Nx+2, Ny+2)
    M_zeropadded[2:end-1, 2:end-1] .= M
    return M_zeropadded
end

#####################################################
# Example utils
#####################################################

function mask_disc(X::KernelMatrix{T}, Y::KernelMatrix{T}, R::T;
    center::Vector{<:Real}) where {T<:AbstractFloat}
    return mask_disc(sqrt.((X .- center[1]) .^ 2 + (Y .- center[2]) .^ 2), R)
end

function mask_disc(r::KernelMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(r .< R)
end

function uniform_ice_cylinder(Omega::ComputationDomain, R::T, H::T;
    center::Vector{T} = T.([0.0, 0.0])) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R, center = center)
    return T.(M .* H)
end

function stereo_ice_cylinder(
    Omega::ComputationDomain,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.R, R)
    return M .* H
end

function stereo_ice_cap(
    Omega::ComputationDomain,
    alpha_deg::T,
    H::T,
) where {T<:AbstractFloat}
    alpha = deg2rad(alpha_deg)
    M = Omega.Theta .< alpha
    return H .* sqrt.( M .* (cos.(Omega.Theta) .- cos(alpha)) ./ (1 - cos(alpha)) )
end