using DelimitedFiles

################################################
# Visualization
################################################

function comparison_figure(n)
    fig = Figure()
    axs = [Axis(fig[i, 1]) for i in 1:n]
    return fig, axs
end

function update_compfig!(axs::Vector{Axis}, fi::Vector, bm::Vector, clr)
    if length(axs) == length(fi) == length(bm)
        nothing
    else
        error("Vectors don't have matching length.")
    end

    for i in eachindex(axs)
        lines!(axs[i], bm[i], color = clr)
        lines!(axs[i], fi[i], color = clr, linestyle = :dash)
    end
end

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
    plotname = "plots/test1/2D/$(case)_Nx$(Omega.Nx)_Ny$(Omega.Ny)"
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
        slicex, slicey = n1÷2:n1, n2÷2
        theta = rad2deg.( Omega.X[slicex, slicey] ./ c.r_equator)

        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_vec .- t) .^ 2 )
            tyr = Int(round( t ))
            lines!(
                axs[i],
                theta,
                U[slicex, slicey, k],
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