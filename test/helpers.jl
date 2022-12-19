
@inline function mask_disc(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(X .^ 2 + Y .^ 2 .< R^2)
end

@inline function generate_uniform_disc_load(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    D = mask_disc(Omega.X, Omega.Y, R)
    return -D .* (c.rho_ice * c.g * H)
end

################################################
# Analytic solution for constant viscosity
################################################
@inline function analytic_solution(
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
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

@inline function looped_quadrature1D( 
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

@inline function analytic_integrand(
    kappa::T,
    r::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}

    beta = mean(p.mantle_density) * c.g + mean(p.lithosphere_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    return (exp(-beta*t/(2*mean(p.halfspace_viscosity)*kappa))-1) * j0 * j1 / beta
end

@inline function equilibrium_integrand(
    kappa::T,
    r::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}
    beta = mean(p.mantle_density) * c.g + mean(p.lithosphere_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    # integrand of inverse Hankel transform when t-->infty
    return - j0 * j1 / beta
end

################################################
# Analytic solution for radially dependent viscosity
################################################

@inline function analytic_radial_solution(
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    H0::T,
    R0::T,
    domains::Vector{T};
    n_quad_support=5::Int,
) where {T<:AbstractFloat}
    scaling = c.rho_ice * c.g * H0 * R0
    radial_integrand(kappa) = analytic_radial_integrand(Omega, i, j, kappa, t, c, p, R0)
    return scaling .* looped_quadrature1D( radial_integrand, domains, n_quad_support )
end

@inline function analytic_radial_integrand(
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    kappa::T,
    t::T,
    c::PhysicalConstants,
    p::SolidEarthParams,
    R0::T,
) where {T<:AbstractFloat}

    x, y = Omega.X[i, j], Omega.Y[i, j]
    r = get_r(x, y)
    # mantle_density = p.mantle_density[i, j]
    # lithosphere_rigidity = p.lithosphere_rigidity[i, j]
    # halfspace_viscosity = p.halfspace_viscosity[i, j]

    # beta = mantle_density * c.g + lithosphere_rigidity * kappa ^ 4
    # j0 = besselj0(kappa * r)
    # j1 = besselj1(kappa * R0)
    # return (exp(-beta*t/(2*halfspace_viscosity*kappa))-1) * j0 * j1 / beta

    beta = mean(p.mantle_density) * c.g + mean(p.lithosphere_rigidity) * kappa ^ 4
    j0 = besselj0(kappa * r)
    j1 = besselj1(kappa * R0)
    return (exp(-beta*t/(2*mean(p.halfspace_viscosity)*kappa))-1) * j0 * j1 / beta
end

################################################
# Visualization
################################################

@inline function plot_response(
    Omega::ComputationDomain,
    sigma::Matrix{T},
    u_plot::Vector{Matrix{T}},
    panels::Vector{Tuple{Int, Int}},
    labels,
    case::String,
) where {T<:AbstractFloat}

    fig = Figure(resolution=(1600, 900))
    ax1 = Axis(fig[1, 1][1, :], aspect=DataAspect())
    hm = heatmap!(ax1, Omega.X, Omega.Y, sigma)
    Colorbar(
        fig[1, 1][2, :],
        hm,
        label = L"Vertical load $ \mathrm{N \, m^{-2}}$",
        vertical = false,
        width = Relative(0.8),
    )

    for k in eachindex(u_plot)
        i, j = panels[k]
        ax3D = Axis3(fig[i, j][1, :])
        sf = surface!(
            ax3D,
            Omega.X,
            Omega.Y,
            u_plot[k],
            # colorrange = (-300, 50),
            colormap = :jet,
        )
        wireframe!(
            ax3D,
            Omega.X,
            Omega.Y,
            u_plot[k],
            linewidth = 0.1,
            color = :black,
        )
        Colorbar(
            fig[i, j][2, :],
            sf,
            label = labels[k],
            vertical = false,
            width = Relative(0.8),
        )
    end
    plotname = "plots/discload_$(case)_N=$(Omega.N)"
    save("$plotname.png", fig)
    save("$plotname.pdf", fig)
    return fig
end

@inline function animate_viscous_response(
    t_vec::AbstractVector{T},
    Omega::ComputationDomain,
    u::Array{T, 3},
    anim_name::String,
    u_range::Tuple{T, T},
    points,
) where {T<:AbstractFloat}

    t_vec = collect(t_vec) ./ (365. * 24. * 60. * 60.)

    # umax = [minimum(u[:, :, j]) for j in axes(u, 3)]
    u_lowest_eta = u[points[1][1], points[1][2], :]
    u_highest_eta = u[points[2][1], points[2][2], :]

    i = Observable(1)
    
    u2D = @lift(u[:, :, $i])
    timepoint = @lift(t_vec[$i])
    upoint_lowest = @lift(u_lowest_eta[$i])
    upoint_highest = @lift(u_highest_eta[$i])

    fig = Figure(resolution = (1600, 900))
    ax1 = Axis(
        fig[1, 1:3],
        xlabel = L"Time $t$ (yr)",
        ylabel = L"Viscous displacement $u^V$ (m)",
        xminorticks = IntervalsBetween(10),
        xminorgridvisible = true,
    )
    ax2 = Axis3(
        fig[1, 5:8],
        xlabel = L"$x$ (m)",
        ylabel = L"$y$ (m)",
        zlabel = L"$z$ (m)",
    )
    cmap = :jet
    clims = u_range

    zlims!(ax2, clims)
    lines!(ax1, t_vec, u_lowest_eta)
    lines!(ax1, t_vec, u_highest_eta)
    scatter!(ax1, timepoint, upoint_lowest, color = :red)
    scatter!(ax1, timepoint, upoint_highest, color = :red)

    sf = surface!(
        ax2,
        Omega.X,
        Omega.Y,
        u2D,
        colorrange = clims,
        colormap = cmap,
    )
    wireframe!(
        ax2,
        Omega.X,
        Omega.Y,
        u2D,
        linewidth = 0.1,
        color = :black,
    )
    Colorbar(
        fig[1, 9],
        sf,
        label = L"Viscous displacement field $u^V$ (m)",
        height = Relative(0.5),
    )

    record(fig, "$anim_name.mp4", axes(u, 3)) do k
        i[] = k
    end
end
