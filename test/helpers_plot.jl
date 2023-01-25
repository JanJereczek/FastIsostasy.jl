
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
            linewidth = 0.01,
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
    plotname = "plots/test1/2D/$(case)_N$(Omega.N)"
    save("$plotname.png", fig)
    save("$plotname.pdf", fig)
    return fig
end

@inline function slice_spada(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    t_vec::AbstractVector{T},
    t_plot::AbstractVector{T},
    Uvec::Vector{Array{T, 3}},
    labels,
    xlabels,
    ylabels,
    plotname::String,
) where {T<:AbstractFloat}

    ncases = length(Uvec)
    fig = Figure(resolution=(1600, 900), fontsize = 24)
    nrows, ncols = 2,2
    axs = [Axis(
        fig[i, j],
        title = labels[(i-1)*2 + j],
        xlabel = xlabels[(i-1)*2 + j],
        ylabel = ylabels[(i-1)*2 + j],
        xminorticks = IntervalsBetween(5),
        yminorticks = IntervalsBetween(2),
        xminorgridvisible = true,
        yminorgridvisible = true,
        xticklabelsvisible = i == 2 ? true : false,
        yticklabelsvisible = j == 1 ? true : false,
    ) for j in 1:2, i in 1:2]
    colors = [:black, :orange, :blue, :red, :gray, :purple]

    for i in 1:ncases
        U = Uvec[i]
        n1, n2, nt = size(U)
        slicey, slicex = Int(round(n1/2)), Int(round(n2/2))
        theta = rad2deg.( Omega.X[slicey, slicex:end] ./ c.r_equator)

        if i == 1
            theta_benchmark = [0, 5, 10, 20]
            scatter_symbols = [:circle, :rect, :diamond]
            for k in eachindex([0, 1, 5, 10, 1000])     # output time vector in spada 2011 (kyr)
                for j in eachindex(["vk", "gs", "zm"])
                    scatter!(
                        axs[i],
                        theta_benchmark,
                        u_benchmark[j, :, k],
                        marker = scatter_symbols[j],
                        color = colors[k],
                    )
                end
            end
        end

        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_vec .- t) .^ 2 )
            tyr = Int(round( seconds2years(t) ))
            lines!(
                axs[i],
                theta,
                U[slicey, slicex:end, k],
                label = L"$t = %$tyr $ yr",
                color = colors[l],
            )
        end
        if i <= 2
            ylims!(axs[i], (-450, 50))
        else
            ylims!(axs[i], (-85, 10))
        end
    end
    axislegend(axs[1], position = :rb)
    save("plots/$plotname.png", fig)
    save("plots/$plotname.pdf", fig)
    return fig
end


@inline function slice_test3(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    t_vec::AbstractVector{T},
    t_plot::AbstractVector{T},
    Uvec::Vector{Array{T, 3}},
    labels,
    xlabels,
    ylabels,
    plotname::String,
) where {T<:AbstractFloat}

    ncases = length(Uvec)
    fig = Figure(resolution=(1600, 600), fontsize = 24)
    nrows, ncols = 1, 3
    axs = [Axis(
        fig[i, j],
        title = labels[(i-1)*ncols + j],
        xlabel = xlabels[(i-1)*ncols + j],
        ylabel = ylabels[(i-1)*ncols + j],
        xminorticks = IntervalsBetween(5),
        yminorticks = IntervalsBetween(2),
        xminorgridvisible = true,
        yminorgridvisible = true,
        xticklabelsvisible = i == nrows ? true : false,
        yticklabelsvisible = j == 1 ? true : false,
    ) for j in 1:ncols, i in 1:nrows]
    colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]

    for i in 1:ncases
        U = Uvec[i]
        n1, n2, nt = size(U)
        slicey, slicex = Int(n1/2), 1:n2
        theta = rad2deg.( Omega.X[slicey, slicex] ./ c.r_equator)

        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_vec .- t) .^ 2 )
            tyr = Int(round( seconds2years(t) ))
            lines!(
                axs[i],
                theta,
                U[slicey, slicex, k],
                label = L"$t = %$tyr $ yr",
                color = colors[l],
            )
        end
        if i <= 3
            ylims!(axs[i], (-550, 50))
        elseif i == 4
            ylims!(axs[i], (-150, 10))
        else
            ylims!(axs[i], (-5000, 1000))
        end
    end
    axislegend(axs[1], position = :rb)
    save("plots/$plotname.png", fig)
    save("plots/$plotname.pdf", fig)
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

u_t0_vk = [-27.81, -23.65, -7.49, -1.20]'
u_t0_gs = [-27.77, -23.64, -7.33, -1.20]'
u_t0_zm = [-27.77, -23.65, -7.38, -1.20]'
u_t0 = vcat(u_t0_vk, u_t0_gs, u_t0_zm)

u_t1_vk = [-94.49, -79.59, -25.25, -1.72]'
u_t1_gs = [-94.40, -79.53, -24.80, -1.72]'
u_t1_zm = [-94.42, -79.55, -24.92, -1.72]'
u_t1 = vcat(u_t1_vk, u_t1_gs, u_t1_zm)

u_t5_vk = [-237.58, -199.62, -48.88, 3.87]'
u_t5_gs = [-237.49, -199.59, -47.73, 3.85]'
u_t5_zm = [-237.50, -199.60, -48.04, 3.85]'
u_t5 = vcat(u_t5_vk, u_t5_gs, u_t5_zm)

u_t10_vk = [-303.03, -256.98, -50.94, 7.07]'
u_t10_gs = [-303.01, -257.05, -49.35, 7.03]'
u_t10_zm = [-302.99, -257.03, -49.77, 7.04]'
u_t10 = vcat(u_t10_vk, u_t10_gs, u_t10_zm)

u_tinf_vk = [NaN, NaN, NaN, NaN]'
u_tinf_gs = [-388.11, -338.30, -59.24, 8.55]'
u_tinf_zm = [NaN, NaN, NaN, NaN]'
u_tinf = vcat(u_tinf_vk, u_tinf_gs, u_tinf_zm)

u_benchmark = cat(u_t0, u_t1, u_t5, u_t10, u_tinf, dims = 3)