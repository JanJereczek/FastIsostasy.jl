push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers/plot.jl")

function main(
    n::Int;
    kernel = "cpu",
)
    N = 2^n
    suffix = "viscmap_N$N"
    sol_het = load("../data/test4/discload_wiens_scaledviscosity_N$N.jld2")
    Omega, p = sol_het["Omega"], sol_het["p"]
    points = [
        CartesianIndex(2^(n-6) * 20, 2^(n-6) * 24),
        CartesianIndex(2^(n-6) * 36, 2^(n-6) * 36),
    ]
    lv = [p.layer_viscosities[:, :, i] for i in axes(p.layer_viscosities, 3)[1:end-1]]
    push!(lv, p.effective_viscosity)

    labels = [L"$\textbf{(%$char)}$" for char in ["a", "b", "c", "d"]]

    xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    visclim = (18, 23)
    viscmap = cgrad(:jet)
    viscticks = (visclim[1]:visclim[2], num2latexstring.(visclim[1]:visclim[2]))

    pcolor = :black
    marker = '⦿'
    yoffset = 2e5

    Xticks = [xticks, xticks, xticks, xticks, xticks]
    Yticks = [yticks, yticks, yticks, yticks, yticks]
    Xticksvisible = [true, true, true, true]
    Yticksvisible = [true, false, false, true]
    xlabels = [
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
    ]
    ylabels = [
        L"$y \: (10^3 \, \mathrm{km})$",
        "",
        "",
        L"$y \: (10^3 \, \mathrm{km})$",
    ]
    Xposition = [:bottom, :bottom, :bottom, :bottom]
    Yposition = [:left, :left, :left, :right]
    fig = Figure(resolution = (1400, 500), fontsize = 20)
    nrows, ncols = 1, 4
    axs = [Axis(
        fig[i, j],
        title = labels[(i-1)*ncols + j],
        titlegap = 10.0,
        xlabel = xlabels[(i-1)*ncols + j],
        ylabel = ylabels[(i-1)*ncols + j],
        xticks = Xticks[(i-1)*ncols + j],
        yticks = Yticks[(i-1)*ncols + j],
        xticklabelsvisible = Xticksvisible[(i-1)*ncols + j],
        yticklabelsvisible = Yticksvisible[(i-1)*ncols + j],
        yticksvisible = Yticksvisible[(i-1)*ncols + j],
        xaxisposition = Xposition[(i-1)*ncols + j],
        yaxisposition = Yposition[(i-1)*ncols + j],
        aspect = AxisAspect(1),
    ) for j in 1:ncols, i in 1:nrows]

    for k in 1:4
        heatmap!(
            axs[k],
            Omega.X,
            Omega.Y,
            log10.(lv[k])',
            colormap = viscmap,
            colorrange = visclim,
        )
        scatter!(
            axs[k],
            [Omega.X[points[1]], Omega.X[points[2]]],
            [Omega.Y[points[1]], Omega.Y[points[2]]],
            color = k <= 3 ? :white : pcolor,
            marker = marker,
            markersize = 20,
        )
        text!(
            axs[k],
            Omega.X[points[1]],
            Omega.Y[points[1]] - yoffset;
            text = L"$\textbf{1}$",
            align = (:center, :top),
            fontsize = 20,
            color = k <= 3 ? :white : pcolor,
        )
        text!(
            axs[k],
            Omega.X[points[2]],
            Omega.Y[points[2]] + yoffset;
            text = L"$\textbf{2}$",
            align = (:center, :bottom),
            fontsize = 20,
            color = k <= 3 ? :white : pcolor,
        )
    end

    Colorbar(
        fig[2, :],
        colormap = viscmap,
        colorrange = visclim,
        vertical = false,
        width = Relative(0.3),
        label = L"Log$_{10}$-viscosity ($\mathrm{Pa \, s})$",
        ticks = viscticks,
    )
    save("plots/$suffix.png", fig)
    save("plots/$suffix.pdf", fig)
end

n = 8
main(n)