push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

function main(
    n::Int;
    kernel = "cpu",
)
    N = 2^n
    suffix = "N$N"
    case = "nonlocal"
    sol_hom = load("data/test4/discload_$(case)_homogeneous_viscosity_$suffix.jld2")
    sol_het = load("data/test4/discload_$(case)_wiens_scaledviscosity_$suffix.jld2")
    res_hom = sol_hom["results"]
    res_het = sol_het["results"]
    t_out = res_het.t_out
    Omega = sol_het["Omega"]
    points = [
        CartesianIndex(2^(n-6) * 20, 2^(n-6) * 24),
        CartesianIndex(2^(n-6) * 36, 2^(n-6) * 36),
    ]

    labels = [W"$\textbf{(%$char)}$" for char in ["a", "b", "c", "d"]]

    xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    tticks = (years2seconds.(0:2e3:10e3), num2latexstring.(0:2:10))
    ulim = (-400, 100)
    umap = cgrad(:cool, rev=true)
    uticks = (ulim[1]:50:ulim[2], num2latexstring.(ulim[1]:50:ulim[2]))
    uticks_sparse = (ulim[1]:100:ulim[2], num2latexstring.(ulim[1]:100:ulim[2]))

    pcolor = :black
    marker = '⦿'
    yoffset = 2e5

    t_plot_hm_yr = [400, 2000, 10000]
    # labels_hm = [W"$t = %$tyr \: \mathrm{yr}$" for tyr in t_plot_hm_yr]
    t_plot_hm = years2seconds.(t_plot_hm_yr)

    Xticks = [xticks, xticks, xticks, tticks]
    Yticks = [yticks, yticks, yticks, uticks]
    Xticksvisible = [true, true, true, true]
    Yticksvisible = [true, false, false, true]
    xlabels = [
        W"$x \: (10^3 \, \mathrm{km})$",
        W"$x \: (10^3 \, \mathrm{km})$",
        W"$x \: (10^3 \, \mathrm{km})$",
        W"$t \: (\mathrm{kyr})$",
    ]
    ylabels = [
        W"$y \: (10^3 \, \mathrm{km})$",
        "",
        "",
        W"$u \: (\mathrm{m})$",
    ]
    Xposition = [:bottom, :bottom, :bottom, :bottom]
    Yposition = [:left, :left, :left, :right]
    fig = Figure(resolution = (1600, 550), fontsize = 20)
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
        xaxisposition = Xposition[(i-1)*ncols + j],
        yaxisposition = Yposition[(i-1)*ncols + j],
        yticksmirrored = true,
        aspect = AxisAspect(1),
    ) for j in 1:ncols, i in 1:nrows]

    u1het = [res_het.viscous[k][points[1]] for k in eachindex(t_out)]
    u2het = [res_het.viscous[k][points[2]] for k in eachindex(t_out)]
    u1hom = [res_hom.viscous[k][points[1]] for k in eachindex(t_out)]
    u2hom = [res_hom.viscous[k][points[2]] for k in eachindex(t_out)]

    for k in eachindex(t_plot_hm)
        t = t_plot_hm[k]
        kt = argmin( (t_out .- t) .^ 2 )
        heatmap!(
            axs[k],
            Omega.X,
            Omega.Y,
            res_het.viscous[kt]',
            colormap = umap,
            colorrange = ulim,
        )

        scatter!(
            axs[k],
            [Omega.X[points[1]], Omega.X[points[2]]],
            [Omega.Y[points[1]], Omega.Y[points[2]]],
            color = pcolor,
            marker = marker,
            markersize = 20,
        )
        text!(
            axs[k],
            Omega.X[points[1]],
            Omega.Y[points[1]] - yoffset;
            text = W"$\textbf{1}$",
            align = (:center, :top),
            fontsize = 20,
            color = k <= 3 ? :white : pcolor,
        )
        text!(
            axs[k],
            Omega.X[points[2]],
            Omega.Y[points[2]] + yoffset;
            text = W"$\textbf{2}$",
            align = (:center, :bottom),
            fontsize = 20,
            color = k <= 3 ? :white : pcolor,
        )
    end

    lines!(axs[ncols], t_out, u1het, label = W"point 1, $\eta(x,y)$")
    lines!(axs[ncols], t_out, u2het, label = W"point 2, $\eta(x,y)$")
    lines!(axs[ncols], t_out, u1hom, label = W"point 1, $\eta(x,y) = c$")
    lines!(axs[ncols], t_out, u2hom, label = W"point 2, $\eta(x,y) = c$")
    vlines!(axs[ncols], t_plot_hm, color = :red)
    axislegend(axs[ncols])
    xlims!(axs[ncols], (years2seconds(-0.5e3), years2seconds(10.5e3)))

    Colorbar(
        fig[2, :],
        colormap = umap,
        colorrange = ulim,
        vertical = false,
        width = Relative(0.3),
        label = W"$u$ (m)",
        ticks = uticks_sparse,
        flipaxis = false,
    )

    save("plots/test4/$(case)_$suffix.png", fig)
    save("plots/test4/$(case)_$suffix.pdf", fig)
end

n = 7
main(n)


# lowest_eta = minimum(p.effective_viscosity[Omega.X.^2 + Omega.Y.^2 .< (1.8e6)^2])
# point_lowest_eta = argmin( (p.effective_viscosity .- lowest_eta).^2 )
# highest_eta = maximum(p.effective_viscosity[Omega.X.^2 + Omega.Y.^2 .< (1.8e6)^2]) 
# point_highest_eta = argmin( (p.effective_viscosity .- highest_eta).^2 )
# points = [point_lowest_eta, point_highest_eta]
# display(points) 
# points = [CartesianIndex(20, 24), CartesianIndex(36, 38)]

# points = sol_het["eta_extrema"]
# CartesianIndex(2 * Omega.N ÷ 5, 2 * Omega.N ÷ 5),
# CartesianIndex(3 * Omega.N ÷ 5, 3 * Omega.N ÷ 5),

# if make_anim
#     anim_name = "plots/test4/discload_$(case)_N$(Omega.N)"
#     animate_viscous_response(
#         t_out,
#         Omega,
#         u3D_viscous,
#         anim_name,
#         (-300.0, 50.0),
#         points,
#         20,
#     )
# end

# if occursin("homogeneous", case) | occursin("meanviscosity", case)
#     checkfig = Figure(resolution = (1600, 700), fontsize = 20)
#     labels = [
#         W"$z \in [88, 400]$ km",
#         W"$z \, > \, 400$ km",
#         W"Equivalent half-space viscosity $\,$",
#     ]
# elseif occursin("scaledviscosity", case)
#     checkfig = Figure(resolution = (1600, 550), fontsize = 20)
#     labels = [
#         W"$z \in [88, 180]$ km",
#         W"$z \in ]180, 280]$ km",
#         W"$z \in ]280, 400]$ km",
#         W"$z \, > \, 400$ km",
#         W"Equivalent half-space viscosity $\,$",
#     ]
# end

# labels = [
#     W"$z \in [-180, -88]$ km",
#     W"$z \in ]-280, -180]$ km",
#     W"$z \in ]-400, -280]$ km",
#     W"$z \, < \, -400$ km",
#     W"Equivalent half-space log-viscosity $\,$",

# ]