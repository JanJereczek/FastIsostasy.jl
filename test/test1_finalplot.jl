push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using SpecialFunctions
using JLD2
using LinearAlgebra
include("helpers_compute.jl")
include("helpers_plot.jl")

function main()

    case = "ExplicitEuler"
    N = 256
    N2 = Int(N/2)
    N4 = Int(N/4)
    hash = "$(case)_N$(N)_gpu_isostate"
    sol = load("data/test1/$hash.jld2")
    Omega, c, p, H, R = sol["Omega"], sol["c"], sol["p"], sol["H"], sol["R"]
    results = sol["results"]

    t_out = results.t_out

    ftsize = 48
    lwidth = 4
    msize = 25
    fig = Figure(resolution = (3300, 1800), fontsize = ftsize)
    t_2Dplot = [500.0, 1500.0, 50_000.0]

    clim = (-300, 10)
    # cmap = cgrad(:PuOr_5, rev = true)
    cmap = cgrad(:cool, rev = true)
    letters = ["a", "b", "c"]
    tgap = 20.0

    for i in eachindex(t_2Dplot)
        tyr = Int(round(t_2Dplot[i]))
        t = years2seconds(t_2Dplot[i])
        k = argmin( (t_out .- t) .^ 2 )
        u_numeric = results.viscous[k]

        letter = letters[i]
        ax3D = Axis3(
            fig[2, i],
            title = W"(%$letter) $t = %$tyr$ yr",
            xlabel = W"$x \: (10^3 \: \mathrm{km})$",
            ylabel = W"$y \: (10^3 \: \mathrm{km})$",
            zlabel = W"$u \: (\mathrm{m})$",
            titlegap = tgap,
            xticks = (-3e6:1e6:3e6, string.(-3:1:3)),
            yticks = (-3e6:1e6:3e6, string.(-3:1:3)),
            xlabeloffset = 80.0,
            ylabeloffset = 80.0,
            zlabeloffset = 130.0,
        )

        surface!(
            ax3D,
            Omega.X,
            Omega.Y,
            u_numeric,
            colorrange = clim,
            colormap = cmap,
        )
        wireframe!(
            ax3D,
            Omega.X,
            Omega.Y,
            u_numeric,
            linewidth = 0.08,
            color = :black,
        )

        zlims!(ax3D, clim)
    end
    Colorbar(
        fig[1, :],
        colorrange = clim,
        colormap = cmap,
        label = W"Viscous displacement $u$ (m)$",
        vertical = false,
        width = Relative(0.4),
        height = 40,
    )

    t_plot = years2seconds.([100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    xoffset = [1, 1, 1, 1, 1, 28]
    yoffset = [10, 10, 10, 10, -40, 0]
    x, y = diag(Omega.X)[N2:N2+N4], diag(Omega.Y)[N2:N2+N4]
    ax3 = Axis(
        fig[3, 1],
        title = W"(d) $\,$",
        xlabel = W"$x \: (10^3 \: \mathrm{km})$ ",
        ylabel = W"$u$ (m)",
        xticks = (0:0.5e6:2e6, string.(0:0.5:2)),
        titlegap = tgap,
    )
    for i in eachindex(t_plot)
        t = t_plot[i]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        k = argmin( (t_out .- t) .^ 2 )
        u_numeric = diag(results.viscous[k])
        tyr = Int(round(seconds2years(t)))

        if i == 1
            lines!(
                ax3,
                x,
                u_numeric[N2:N2+N4],
                label = W"numeric $\,$",
                color = colors[i],
                linewidth = lwidth,
            )
            lines!(
                ax3,
                x,
                u_analytic,
                label = W"analytic $\,$",
                linestyle = :dash,
                color = colors[i],
                linewidth = lwidth,
            )
        else
            lines!(
                ax3,
                x,
                u_numeric[N2:N2+N4],
                color = colors[i],
                linewidth = lwidth,
            )
            lines!(
                ax3,
                x,
                u_analytic,
                linestyle = :dash,
                color = colors[i],
                linewidth = lwidth,
            )
        end

        text!(
            ax3,
            x[xoffset[i]],
            u_numeric[N2] + yoffset[i],
            text = W"$ %$tyr $ yr",
            align = (:left, :bottom),
            color = colors[i],
            fontsize = ftsize,
        )
    end
    axislegend(ax3, position = :rc, width = 280, linepoints = [Point2f(0, 0.5), Point2f(2.5, 0.5)], patchlabelgap = 40)

    Nvec = 2 .^ (4:8)
    maxerror = Float64[]
    meanerror = Float64[]
    delta_x = Float64[]
    t_end = years2seconds(1e5)

    for N in Nvec
        hash = "$(case)_N$(N)_gpu_isostate"
        sol = load("data/test1/$hash.jld2")

        islice, jslice = Int(round(N/2)), Int(round(N/2))
        Omega, c, p, H, R = sol["Omega"], sol["c"], sol["p"], sol["H"], sol["R"]
        results = sol["results"]
        x, y = Omega.X[islice, jslice:end], Omega.Y[islice, jslice:end] 
        analytic_solution_r(r) = analytic_solution(r, t_end, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        u_numeric = results.viscous[end]
        abs_error = abs.(u_analytic - u_numeric[islice, jslice:end])

        append!(delta_x, Omega.Wx * 1e-3 / N)
        append!(maxerror, maximum(abs_error))
        append!(meanerror, mean(abs_error))
    end
    ax1 = Axis(
        fig[3,2],
        title = W"(e) $\,$",
        xlabel = W"$N = N_{x} = N_{y} $ (1)",
        ylabel = W"Error (m)$\,$",
        xscale = log2,
        yscale = log10,
        yticks = (10. .^ (-1:1), [W"$10^{%$l}$" for l in -1:1]),
        yminorticks = IntervalsBetween(9),
        yminorticksvisible = true,
        yminorgridvisible = true,
        titlegap = tgap,
    )

    scatterlines!(ax1, Nvec, maxerror, label = W"Maximum $\,$", linewidth = lwidth, markersize = msize)
    scatterlines!(ax1, Nvec, meanerror, label = W"Average $\,$", linewidth = lwidth, markersize = msize)
    axislegend(ax1, position = :lb, width = 320, linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)

    ax2 = Axis(
        fig[3,3],
        title = W"(f) $\,$",
        xlabel = W"$N = N_{x} = N_{y} $ (1)",
        ylabel = W"Run time (s) $\,$",
        xscale = log2,
        yscale = log10,
        yticks = (10. .^ (0:3), [W"$10^{%$l} $" for l in 0:3]),
        yminorticks = IntervalsBetween(9),
        yminorticksvisible = true,
        yminorgridvisible = true,
        titlegap = tgap,
    )
    for kernel in ["cpu", "gpu"]
        runtime = Float64[]
        delta_x = Float64[]
        for N in Nvec
            hash = "$(case)_N$(N)_$(kernel)_isostate"
            sol = load("data/test1/$hash.jld2")
            append!(runtime, sol["t_fastiso"])
            append!(delta_x, 2*sol["Omega"].Wx * 1e-3 / N)
        end
        scatterlines!(ax2, Nvec, runtime, label = W"%$kernel $\,$", linewidth = lwidth, markersize = msize)
    end
    axislegend(ax2, position = :lt, width = 200, linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 80)

    save("plots/test1/finalplot.png", fig)
    save("plots/test1/finalplot.pdf", fig)
end

main()