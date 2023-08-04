#####################################################
# Data loaders
#####################################################
function interpolate_spada_benchmark(c, data)
    idx = sortperm(data[:, 1])
    theta = data[:, 1][idx]
    z = data[:, 2][idx]
    x = deg2rad.(theta) .* c.r_equator
    itp = linear_interpolation(x, z, extrapolation_bc = Flat())
    return itp
end

function load_spada()
    prefix ="../data/test2/Spada/"
    cases = ["u_cap", "u_disc", "dudt_cap", "dudt_disc", "n_cap", "n_disc"]
    snapshots = ["0", "1", "2", "5", "10", "inf"]
    data = Dict{String, Vector{Matrix{Float64}}}()
    for case in cases
        tmp = Matrix{Float64}[]
        for snapshot in snapshots
            fname = string(prefix, case, "_", snapshot, ".csv")
            append!(tmp, [readdlm(fname, ',', Float64)])
        end
        data[case] = tmp
    end
    return data
end

function load_latychev(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]

    u = zeros(length(x), length(files))
    for i in eachindex(files)
        file = files[i]
        # println( file, typeof( readdlm(joinpath(dir, file), ',')[:, 1] ) )
        u[:, i] = readdlm(joinpath(dir, file), ',')[idx, 2]
    end
    u .-= u[:, 1]

    return x, u
end

#####################################################
# Idealised load cases
#####################################################
function mask_disc(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return mask_disc(sqrt.(X.^2 + Y.^2), R)
end

function mask_disc(r::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(r .< R)
end

function uniform_ice_cylinder(
    Omega::ComputationDomain,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R)
    return M .* H
end

function stereo_ice_cylinder(
    Omega::ComputationDomain,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.R, R)
    return M .* H
end

function generate_uniform_disc_load(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R)
    return - M .* (c.rho_ice * c.g * H)
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

################################################
# Analytic solution for constant viscosity
################################################
function analytic_solution(
    r::T,
    t::T,
    c::PhysicalConstants,
    p::LateralVariability,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=5::Int,
) where {T<:AbstractFloat}
    scaling = c.rho_ice * c.g * H0 * R0
    if t == T(Inf)
        equilibrium_integrand_r(kappa) = equilibrium_integrand(kappa, r, c, p, R0)
        return scaling .* looped_quadrature1D( equilibrium_integrand_r, domains, n_quad_support )
    else
        transient_integrand_r(kappa) = analytic_integrand(kappa, r, t, c, p, R0)
        return scaling .* looped_quadrature1D( transient_integrand_r, domains, n_quad_support )
    end
end

function looped_quadrature1D( 
    f::Function,
    domains::Vector{T},
    n::Int,
) where{T<:Real}
    integral = T(0)
    for i in eachindex(domains)[1:end-1]
        integral += quadrature1D( f, n, domains[i], domains[i+1] )
    end
    return integral
end

function analytic_integrand(
    kappa::T,
    r::T,
    t::T,
    c::PhysicalConstants,
    p::LateralVariability,
    R0::T,
) where {T<:AbstractFloat}

    # Here we assume that p-fields are constant over Omega
    beta = c.rho_uppermantle * c.g + mean(p.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    eta = mean(p.effective_viscosity)
    return (exp(-beta*t/(2*eta*kappa))-1) * j0 * j1 / beta
end

function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    p::LateralVariability,
    R0::T,
) where {T<:AbstractFloat}
    beta = c.rho_uppermantle * c.g + mean(p.litho_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end

################################################
# Generate binary parameter fields for test 3
################################################

function generate_gaussian_field(
    Omega::ComputationDomain{T},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::AbstractMatrix{T},
) where {T<:AbstractFloat}
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

function slice_along_x(Omega::ComputationDomain)
    Nx, Ny = Omega.Nx, Omega.Ny
    return Nx÷2:Nx, Ny÷2
end
