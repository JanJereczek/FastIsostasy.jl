using DelimitedFiles

################################################
# Visualization
################################################

function num2latexstring(x::Real)
    return L"%$x"
end

function string2latexstring(x::String)
    return L"%$x $\,$"
end

function plot_response(
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

function slice_test3(
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

function animate_viscous_response(
    t_vec::AbstractVector{T},
    Omega::ComputationDomain,
    u::Array{T, 3},
    anim_name::String,
    u_range::Tuple{T, T},
    points,
    framerate::Int,
) where {T<:AbstractFloat}

    t_vec = seconds2years.(t_vec)

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
        zlabel = L"$u^V$ (m)",
    )
    cmap = cgrad(:cool, rev = true)
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
        linewidth = 0.08,
        color = :black,
    )
    Colorbar(
        fig[1, 9],
        sf,
        label = L"Viscous displacement field $u^V$ (m)",
        height = Relative(0.5),
    )

    record(fig, "$anim_name.mp4", axes(u, 3), framerate = framerate) do k
        i[] = k
    end
end
