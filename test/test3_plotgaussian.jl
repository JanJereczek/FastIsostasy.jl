push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

@inline function main(
    n::Int;             # 2^n cells on domain (1)
    make_anim = false,
    kernel = "gpu",
)

    N = 2^n
    suffix = "$(kernel)_N$N"
    sol_lo_D = load("data/test3/gaussian_lo_D_$suffix.jld2")
    sol_hi_D = load("data/test3/gaussian_hi_D_$suffix.jld2")
    sol_lo_η = load("data/test3/gaussian_lo_η_$suffix.jld2")
    sol_hi_η = load("data/test3/gaussian_hi_η_$suffix.jld2")

    sols = [sol_lo_D, sol_hi_D, sol_lo_η, sol_hi_η]
    u_plot = [sol["u3D_viscous"] for sol in sols]
    D_plot = [sol["p"].litho_thickness for sol in sols[1:2]]
    η_plot = [log10.(sol["p"].layers_viscosity[:,:,1]) for sol in sols[3:4]]
    var_plots = u_plot
    p_plots = vcat(D_plot, η_plot)

    labels = [
        L"Small lithospheric thickness $T$",
        L"Large lithospheric thickness $T$",
        L"Low viscosity $\eta$",
        L"High viscosity $\eta$",
        "",
        "",
        "",
        "",
    ]

    xlabels = [
        L"$x$ (km)",
        L"$x$ (km)",
        L"$x$ (km)",
        L"$x$ (km)",
        L"Colatitude $\theta$ (deg)",
        L"Colatitude $\theta$ (deg)",
        L"Colatitude $\theta$ (deg)",
        L"Colatitude $\theta$ (deg)",
    ]

    ylabels = [
        L"$y$ (km)",
        "",
        "",
        "",
        L"Displacement $u \: (\mathrm{m})$",
        "",
        "",
        "",
    ]

    t_plot_yr = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
    t_plot = years2seconds.(t_plot_yr)
    plotname = "test3/$suffix"
    Omega, c, t_out = sol_lo_D["Omega"], sol_lo_D["c"], sol_lo_D["t_out"]

    ncases = length(var_plots)
    fig = Figure(resolution=(1600, 900), fontsize = 24)
    nrows, ncols = 2, 4
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
    cmaps = [cgrad(:RdBu, rev=true), cgrad(:RdBu, rev=true),
        cgrad(:PuOr, rev=true), cgrad(:PuOr, rev=true)]
    clims = [(50e3, 250e3), (50e3, 250e3), (20, 22), (20, 22)]

    for j in 1:ncases

        hm = heatmap!(
            axs[j],
            p_plots[j],
            colormap = cmaps[j],
            colorrange = clims[j],
        )
        hidedecorations!(axs[j])
        if rem(j, 2) == 0
            k = Int(j/2)
            Colorbar(
                fig[3, 2*k-1:2*k],
                hm,
                vertical = false,
                width = Relative(0.6),
            )
        end

        i = j + ncols

        varplot = var_plots[j]
        n1, n2, nt = size(varplot)
        slicey, slicex = Int(n1/2), 1:n2
        theta = rad2deg.( Omega.X[slicey, slicex] ./ c.r_equator)

        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_out .- t) .^ 2 )
            tyr = Int(round( seconds2years(t) ))
            lines!(
                axs[i],
                theta,
                varplot[slicey, slicex, k],
                label = L"$%$tyr $ yr",
                color = colors[l],
            )
        end
        ylims!(axs[i], (-600, 50))
    end
    axislegend(axs[5], position = :rb, nbanks=2)
    save("plots/$(plotname)_gaussian.png", fig)
    save("plots/$(plotname)_gaussian.pdf", fig)

end

for n in 5:5
    main(n, kernel = "cpu")
end