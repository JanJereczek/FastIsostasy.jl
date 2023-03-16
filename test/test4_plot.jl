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
    sol_hom = load("data/test4/discload_homogeneous_viscosity_$suffix.jld2")
    sol_het = load("data/test4/discload_wiens_scaledviscosity_$suffix.jld2")
    res_hom = sol_hom["results"]
    res_het = sol_het["results"]
    t_out = res_het.t_out
    Omega, p = sol_het["Omega"], sol_het["p"]
    # points = sol_het["eta_extrema"]
    points = [
        CartesianIndex(2 * Omega.N ÷ 5, 2 * Omega.N ÷ 5),
        CartesianIndex(3 * Omega.N ÷ 5, 3 * Omega.N ÷ 5),
    ]
    lv = [p.layers_viscosity[:, :, i] for i in axes(p.layers_viscosity, 3)]
    push!(lv, p.effective_viscosity)

    labels = ([
        L"\textbf{(a)} $\,$", L"\textbf{(b)} $\,$", L"\textbf{(c)} $\,$",
        L"\textbf{(d)} $\,$", L"\textbf{(e)} $\,$", L"\textbf{(f)} $\,$",
        L"\textbf{(g)} $\,$", L"\textbf{(h)} $\,$", L"\textbf{(i)} $\,$",
    ])

    xticks = (-4e6:1e6:4e6, num2latexstring.(-4:4))
    yticks = (-4e6:1e6:4e6, num2latexstring.(-4:4))
    tticks = (years2seconds.(0:5e3:20e3), num2latexstring.(0:5:20))
    ulim = (-350, 50)
    umap = cgrad(:cool, rev=true)
    uticks = (ulim[1]:50:ulim[2], num2latexstring.(ulim[1]:50:ulim[2]))
    visclim = (18, 23)
    viscmap = cgrad(:jet)
    viscticks = (visclim[1]:visclim[2], num2latexstring.(visclim[1]:visclim[2]))

    t_plot_hm_yr = [200, 2000, 20000]
    # labels_hm = [L"$t = %$tyr \: \mathrm{yr}$" for tyr in t_plot_hm_yr]
    t_plot_hm = years2seconds.(t_plot_hm_yr)

    Xticks = [
        xticks, xticks, xticks, xticks, xticks,
        tticks,
        xticks, xticks, xticks,
    ]
    Yticks = [
        yticks, yticks, yticks, yticks, yticks,
        uticks,
        yticks, yticks, yticks,
    ]
    Xticksvisible = [true, true, true, false, false, true, true, true, true]
    Yticksvisible = [true, false, true, true, false, true, true, false, true]
    xlabels = [
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
        "",
        "",
        L"$t \: (\mathrm{kyr})$",
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
        L"$x \: (10^3 \, \mathrm{km})$",
    ]
    ylabels = [
        L"$y \: (10^3 \, \mathrm{km})$",
        "",
        L"$y \: (10^3 \, \mathrm{km})$",
        L"$y \: (10^3 \, \mathrm{km})$",
        "",
        L"Displacement $u \: (\mathrm{m})$",
        L"$y \: (10^3 \, \mathrm{km})$",
        "",
        L"$y \: (10^3 \, \mathrm{km})$",
    ]
    Xposition = [:top, :top, :top, :bottom, :bottom, :bottom, :bottom, :bottom, :bottom]
    Yposition = [:left, :left, :right, :left, :left, :right, :left, :left, :right]
    fig = Figure(resolution = (1500, 1500), fontsize = 20)
    nrows, ncols = 3, 3
    axs = [Axis(
        fig[i+1, j],
        title = labels[(i-1)*ncols + j],
        xlabel = xlabels[(i-1)*ncols + j],
        ylabel = ylabels[(i-1)*ncols + j],
        xticks = Xticks[(i-1)*ncols + j],
        yticks = Yticks[(i-1)*ncols + j],
        xticklabelsvisible = Xticksvisible[(i-1)*ncols + j],
        yticklabelsvisible = Yticksvisible[(i-1)*ncols + j],
        xaxisposition = Xposition[(i-1)*ncols + j],
        yaxisposition = Yposition[(i-1)*ncols + j],
        xticksmirrored = true,
        yticksmirrored = true,
    ) for j in 1:ncols, i in 1:nrows]
    fig

    for k in 1:5
        heatmap!(
            axs[k],
            Omega.X,
            Omega.Y,
            log10.(lv[k]),
            colormap = viscmap,
            colorrange = visclim,
        )
        scatter!(
            axs[k],
            [Omega.X[points[1]], Omega.X[points[2]]],
            [Omega.Y[points[1]], Omega.Y[points[2]]],
            color = :white,
            markersize = 20,
        )
    end

    u1het = [res_het.viscous[k][points[1]] for k in eachindex(t_out)]
    u2het = [res_het.viscous[k][points[2]] for k in eachindex(t_out)]
    u1hom = [res_hom.viscous[k][points[1]] for k in eachindex(t_out)]
    u2hom = [res_hom.viscous[k][points[2]] for k in eachindex(t_out)]
    lines!(axs[6], t_out, u1het, label = L"point 1, $\eta(x,y)$")
    lines!(axs[6], t_out, u2het, label = L"point 2, $\eta(x,y)$")
    lines!(axs[6], t_out, u1hom, label = L"point 1, $\eta(x,y) = \eta$")
    lines!(axs[6], t_out, u2hom, label = L"point 2, $\eta(x,y) = \eta$")
    vlines!(axs[6], t_plot_hm, color = :red)
    axislegend(axs[6])
    # xlims!(axs[6], (0, years2seconds(2e3)))

    for k in 7:9
        t = t_plot_hm[k-6]
        kt = argmin( (t_out .- t) .^ 2 )
        heatmap!(
            axs[k],
            Omega.X,
            Omega.Y,
            res_het.viscous[kt],
            colormap = umap,
            colorrange = ulim,
        )

        scatter!(
            axs[k],
            [Omega.X[points[1]], Omega.X[points[2]]],
            [Omega.Y[points[1]], Omega.Y[points[2]]],
            color = :white,
            markersize = 20,
        )
    end

    Colorbar(
        fig[1, :],
        colormap = viscmap,
        colorrange = visclim,
        vertical = false,
        width = Relative(0.3),
        label = L"log-viscosity ($\mathrm{Pa \, s})$",
        ticks = viscticks,
    )
    Colorbar(
        fig[5, :],
        colormap = umap,
        colorrange = ulim,
        vertical = false,
        width = Relative(0.3),
        label = L"vertical displacement (m) $\,$",
        ticks = uticks,
    )

    save("plots/test4/$suffix.png", fig)
    save("plots/test4/$suffix.pdf", fig)
end

main(6)

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
#         L"$z \in [88, 400]$ km",
#         L"$z \, > \, 400$ km",
#         L"Equivalent half-space viscosity $\,$",
#     ]
# elseif occursin("scaledviscosity", case)
#     checkfig = Figure(resolution = (1600, 550), fontsize = 20)
#     labels = [
#         L"$z \in [88, 180]$ km",
#         L"$z \in ]180, 280]$ km",
#         L"$z \in ]280, 400]$ km",
#         L"$z \, > \, 400$ km",
#         L"Equivalent half-space viscosity $\,$",
#     ]
# end

# labels = [
#     L"$z \in [-180, -88]$ km",
#     L"$z \in ]-280, -180]$ km",
#     L"$z \in ]-400, -280]$ km",
#     L"$z \, < \, -400$ km",
#     L"Equivalent half-space log-viscosity $\,$",

# ]