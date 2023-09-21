push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie, SpecialFunctions, JLD2, LinearAlgebra
include("../../test/helpers/compute.jl")
include("../../test/helpers/plot.jl")
include("../helpers.jl")

function main()

    H, R = 1e3, 1000e3
    N = 256
    filename = "Nx=$(N)_Ny=$(N)_cpu_interactive_sl=false"
    @load "../data/test1/$filename.jld2" fip Hice
    Omega, c, p, t_out = fip.Omega, fip.c, fip.p, fip.out.t

    ftsize = 50
    lwidth = 5
    msize = 35
    fig = Figure(resolution = (1800, 1800), fontsize = ftsize)

    tgap = 20.0
    ii, jj = slice_along_x(Omega)
    x, y = Omega.X[ii, jj], Omega.Y[ii, jj]

    t_plot_yr = [100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0]
    t_plot = years2seconds.(t_plot_yr)
    # colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]
    cmap = cgrad(janjet, length(t_plot), categorical = true)

    # cmap = cgrad(:darkrainbow, length(t_plot), categorical = true)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    xoffset = [2, 2, 2, 2, 2, 40]
    yoffset = [10, 8, 9, 10, -30, 0]

    ax3 = Axis(fig[1, 1], title = L"(a) $\,$", xlabel = L"$x \: (10^3 \: \mathrm{km})$ ",
        ylabel = L"$u$ (m)", xticks = (0:0.5e6:2e6, latexify(0:0.5:2)), titlegap = tgap,
        yticks = latexticks(-300:100:0))
    maxerror = fill(Inf, length(t_plot))
    meanerror = fill(Inf, length(t_plot))
    for i in eachindex(t_plot)
        t = t_plot[i]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        k = argmin( (t_out .- t) .^ 2 )
        u_numeric = fip.out.u[k][ii, jj]
        tyr = Int(round(seconds2years(t)))
        abs_error = abs.(u_analytic - u_numeric)
        lines!(ax3, x, u_numeric, color = cmap[i], linewidth = lwidth)
        lines!(ax3, x, u_analytic, linestyle = :dash, color = cmap[i],
            linewidth = lwidth)


        text!(
            ax3,
            x[xoffset[i]],
            u_numeric[1] + yoffset[i],
            text = L"$ %$tyr $ yr",
            align = (:left, :bottom),
            color = cmap[i],
            fontsize = ftsize,
        )
        maxerror[i] = maximum(abs_error)
        meanerror[i] = mean(abs_error)
    end
    # General legend
    lines!(ax3, x, fill(1e8, length(x)), label = L"numeric $\,$", color = :gray20,
        linewidth = lwidth)
    lines!(ax3, x, fill(1e8, length(x)), label = L"analytic $\,$", color = :gray20,
        linewidth = lwidth, linestyle = :dash)
    axislegend(ax3, position = :rc, width = 280,
        linepoints = [Point2f(0, 0.5), Point2f(2.5, 0.5)], patchlabelgap = 40)
    xlims!(ax3, (0, 1.8e6))
    ylims!(ax3, (-290, 30))

    
    ax4 = Axis(fig[2, 1], title = L"(c) $\,$", xlabel = L"Time (kyr) $\,$",
        ylabel = L"Absolute error (m) $\,$", titlegap = tgap,
        xticks = (eachindex(t_plot), latexify(round.(t_plot_yr ./ 1e3, digits = 1))),
        yticks = latexticks(0:2:10))
    ylims!(ax4, (0, 10))
    barplot!(ax4, eachindex(t_plot), maxerror, label = L"max $\,$")
    barplot!(ax4, eachindex(t_plot), meanerror, label = L"mean $\,$")
    axislegend(ax4, position = :rt)

    Nvec = 2 .^ (4:8)
    maxerror = fill(Inf, length(Nvec))
    meanerror = fill(Inf, length(Nvec))
    delta_x = fill(Inf, length(Nvec))
    t_end = years2seconds(t_plot[end])
    for n in eachindex(Nvec)
        Ni = Nvec[n]
        fname = "Nx=$(Ni)_Ny=$(Ni)_cpu_interactive_sl=false"
        @load "../data/test1/$fname.jld2" fip
        Omega, c, p, t_out = fip.Omega, fip.c, fip.p, fip.out.t

        ii, jj = slice_along_x(Omega)
        x, y = Omega.X[ii, jj], Omega.Y[ii, jj]
        analytic_solution_r(r) = analytic_solution(r, t_end, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        u_numeric = fip.out.u[end][ii, jj]
        abs_error = abs.(u_analytic - u_numeric)

        delta_x[n] = Omega.Wx * 1e-3 / N
        maxerror[n] = maximum(abs_error)
        meanerror[n] = mean(abs_error)
    end
    ax5 = Axis(fig[1, 2], title = L"(b) $\,$", xlabel = L"$N = N_{x} = N_{y} $ (1)",
        ylabel = L"Absolute error (m)$\,$", xscale = log2, yscale = log10,
        xticks = (2 .^ (4:8), [L"$2^{%$l}$" for l in 4:8]),
        yticks = (10. .^ (-1:1), [L"$10^{%$l}$" for l in -1:1]),
        yminorticks = IntervalsBetween(9),
        yminorticksvisible = true,
        yminorgridvisible = true,
        titlegap = tgap,
        yaxisposition = :right,
    )

    scatterlines!(ax5, Nvec, maxerror, label = L"max $\,$", linewidth = lwidth, markersize = msize)
    scatterlines!(ax5, Nvec, meanerror, label = L"mean $\,$", linewidth = lwidth, markersize = msize)
    axislegend(ax5, position = :lb, width = 320,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
    ylims!(ax5, (1e-1, 1e2))
    # barplot!(ax5, Nvec, maxerror, label = L"max $\,$", linewidth = lwidth, markersize = msize)
    # barplot!(ax5, Nvec, meanerror, label = L"mean $\,$", linewidth = lwidth, markersize = msize)
    # axislegend(ax5, position = :rt, width = 320)

    ax6 = Axis(
        fig[2, 2],
        title = L"(d) $\,$",
        xlabel = L"$N = N_{x} = N_{y} $ (1)",
        ylabel = L"Run time (s) $\,$",
        xscale = log2,
        yscale = log10,
        xticks = (2 .^ (4:8), [L"$2^{%$l}$" for l in 4:8]),
        yticks = (10. .^ (0:3), [L"$10^{%$l} $" for l in 0:3]),
        yminorticks = IntervalsBetween(9),
        yminorticksvisible = true,
        yminorgridvisible = true,
        titlegap = tgap,
        yaxisposition = :right,
    )
    for kernel in ["cpu", "gpu"]
        runtime = Float64[]
        delta_x = Float64[]
        for N in Nvec
            fname = "Nx=$(N)_Ny=$(N)_$(kernel)_interactive_sl=false"
            @load "../data/test1/$fname.jld2" fip

            append!(runtime, fip.out.computation_time)
            append!(delta_x, 2*fip.Omega.Wx * 1e-3 / N)
        end
        scatterlines!(ax6, Nvec, runtime, label = L"%$kernel $\,$", linewidth = lwidth, markersize = msize)
    end
    axislegend(ax6, position = :lt, width = 200,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)

    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)

    save("plots/test1/finalplot_v0.3.png", fig)
    save("plots/test1/finalplot_v0.3.pdf", fig)
end

main()