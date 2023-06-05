push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

function main(
    n::Int;             # 2^n cells on domain (1)
    kernel = "cpu",
)

    N = 2^n
    suffix = "$(kernel)_N$N"
    sol_lo_D = load("data/test3/gaussian_lo_D_$suffix.jld2")
    sol_hi_D = load("data/test3/gaussian_hi_D_$suffix.jld2")
    sol_lo_η = load("data/test3/gaussian_lo_η_$suffix.jld2")
    sol_hi_η = load("data/test3/gaussian_hi_η_$suffix.jld2")

    sols = [sol_lo_D, sol_hi_D, sol_lo_η, sol_hi_η]
    results = [sol["results"] for sol in sols]
    u_plot = [res.viscous for res in results]
    D_plot = [sol["p"].litho_thickness for sol in sols[1:2]]
    η_plot = [log10.(sol["p"].layer_viscosities[:,:,1]) for sol in sols[3:4]]
    var_plots = u_plot
    p_plots = vcat(D_plot, η_plot)

    labels = [
        W"(a) $\,$", # W"Small lithospheric thickness $T$",
        W"(b) $\,$", # W"Large lithospheric thickness $T$",
        W"(c) $\,$", # W"Low upper-mantle viscosity $\eta$",
        W"(d) $\,$", # W"High upper-mantle viscosity $\eta$",
        "",
        "",
        "",
        "",
    ]

    xlabels = [
        "",
        "",
        "",
        "",
        W"$x \: (10^3 \, \mathrm{km})$",
        W"$x \: (10^3 \, \mathrm{km})$",
        W"$x \: (10^3 \, \mathrm{km})$",
        W"$x \: (10^3 \, \mathrm{km})$",
    ]

    ylabels = [
        W"$y \: (10^3 \, \mathrm{km})$",
        "",
        "",
        "",
        W"$u \: (\mathrm{m})$",
        "",
        "",
        "",
    ]
    xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    ulim = (-350, 50)
    uticks = (ulim[1]:50:ulim[2], num2latexstring.(ulim[1]:50:ulim[2]))

    logt_plot = collect(0:5)
    texlabels = [W"$10^{%$t}$ years" for t in logt_plot]
    t_plot = years2seconds.(10 .^ logt_plot)
    plotname = "test3/$suffix"
    Omega, c = sol_lo_D["Omega"], sol_lo_D["c"]
    t_out = results[1].t_out

    ncases = length(var_plots)
    fig = Figure(resolution=(1600, 900), fontsize = 24)
    nrows, ncols = 2, 4
    axs = [Axis(
        fig[i+1, j],
        title = labels[(i-1)*ncols + j],
        xlabel = xlabels[(i-1)*ncols + j],
        ylabel = ylabels[(i-1)*ncols + j],
        xticks = xticks,
        yticks = i==1 ? yticks : uticks,
        # xminorticks = IntervalsBetween(5),
        # yminorticks = IntervalsBetween(2),
        # xminorgridvisible = true,
        # yminorgridvisible = true,
        xticklabelsvisible = i == nrows ? true : false,
        yticklabelsvisible = j == 1 ? true : false,
        aspect = AxisAspect(1),
    ) for j in 1:ncols, i in 1:nrows]
    colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]
    rigidity_map = cgrad([:royalblue3, :white, :red3])
    viscosity_map = cgrad([:purple3, :white, :orange])
    cmaps = [rigidity_map, rigidity_map, viscosity_map, viscosity_map]
    # cmaps = [cgrad(:RdBu, rev=true), cgrad(:RdBu, rev=true),
    #    cgrad(:PuOr, rev=true), cgrad(:PuOr, rev=true)]

    clims = [(50e3, 250e3), (50e3, 250e3), (19, 23), (19, 23)]
    lins = Lines[]
    cticks = [
        (50e3:50e3:250e3, num2latexstring.(50:50:250)),
        (19:23, num2latexstring.(19:23)),
    ]
    clabels = [W"Lithospheric thickness (km) $\,$", W"Upper-mantle log-viscosity ($\mathrm{Pa \, s}$)"]

    for j in 1:ncases

        hm = heatmap!(
            axs[j],
            Omega.X,
            Omega.Y,
            p_plots[j],
            colormap = cmaps[j],
            colorrange = clims[j],
        )
        # hidedecorations!(axs[j])
        if rem(j, 2) == 0
            k = Int(j/2)
            Colorbar(
                fig[1, 2*k-1:2*k],
                hm,
                vertical = false,
                width = Relative(0.6),
                ticks = cticks[k],
                label = clabels[k],
            )
        end

        i = j + ncols

        varplot = var_plots[j]
        nt = length(varplot)
        n1, n2 = size(varplot[1])
        slicey, slicex = Int(n1/2), 1:n2
        x = Omega.X[slicey, slicex]
        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_out .- t) .^ 2 )
            tyr = Int(round( seconds2years(t) ))
            lin = lines!(
                axs[i],
                x,
                varplot[k][slicey, slicex],
                label = W"$%$tyr $ yr",
                color = colors[l],
            )
            if i == ncols + 1
                push!(lins, lin)
            end
        end
        ylims!(axs[i], ulim)
    end
    # Legend(fig[4, :], lins, texlabels, nbanks=6)
    Legend(fig[3, 5], lins, texlabels, nbanks=1)
    save("plots/$(plotname)_gaussian.png", fig)
    save("plots/$(plotname)_gaussian.pdf", fig)

end

for n in 7:7
    main(n, kernel = "cpu")
end