using FastIsostasy
using CairoMakie, SpecialFunctions, JLD2, LinearAlgebra
include("../../test/helpers/compute.jl")
include("../../test/helpers/plot.jl")
include("../helpers.jl")

function linregression(x, y)
    X = hcat(x, ones(length(x)))
    return inv(X' * X) * X' * y
end

function main()

    H, R = 1e3, 1000e3
    N = 256
    filename = "nx=$(N)_ny=$(N)_cpu_interactive_sl=false-dense"
    @load "../data/test1/$filename.jld2" fip Hice
    Omega, c, p, t_out = fip.Omega, fip.c, fip.p, fip.out.t

    ftsize = 54
    lwidth = 7
    msize = 35
    fig = Figure(size = (1800, 1800), fontsize = ftsize)

    tgap = 20.0
    ii, jj = slice_along_x(Omega)
    x, y = Omega.X[ii, jj], Omega.Y[ii, jj]

    t_plot_yr = [100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0]
    t_plot = years2seconds.(t_plot_yr)
    # colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]
    cmap = cgrad(janjet, length(t_plot), categorical = true)

    # cmap = cgrad(:darkrainbow, length(t_plot), categorical = true)
    xoffset = [2, 2, 2, 2, 2, 40]
    yoffset = [9, 7, 9, 11, 11, 0]

    ax3 = Axis(fig[1, 1], title = L"(a) $\,$", xlabel = L"$x \: (10^3 \: \mathrm{km})$ ",
        ylabel = L"$u$ (m)",
        xticks = (0:0.5e6:1.5e6, latexify(Union{Float64, Int}[0, 0.5, 1, 1.5])),
        titlegap = tgap,
        yticks = latexticks(-300:100:0)
    )
    maxerror_t = fill(Inf, length(t_plot))
    meanerror_t = fill(Inf, length(t_plot))
    for i in eachindex(t_plot)
        t = t_plot[i]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R)
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
        maxerror_t[i] = maximum(abs_error)
        meanerror_t[i] = mean(abs_error)
    end
    # General legend
    lines!(ax3, x, fill(1e8, length(x)), label = L"numeric $\,$", color = :gray20,
        linewidth = lwidth)
    lines!(ax3, x, fill(1e8, length(x)), label = L"analytic $\,$", color = :gray20,
        linewidth = lwidth, linestyle = :dash)
    axislegend(ax3, position = :rc, width = 300,
        linepoints = [Point2f(0, 0.5), Point2f(2.5, 0.5)], patchlabelgap = 40)
    xlims!(ax3, (0, 1.8e6))
    ylims!(ax3, (-290, 30))


    ax4 = Axis(fig[1, 2], title = L"(b) $\,$", xlabel = L"Time (kyr) $\,$",
        ylabel = L"Absolute error (m) $\,$", titlegap = tgap,
        # xticks = (eachindex(t_plot), latexify(round.(t_plot_yr ./ 1e3, digits = 1))),
        xticks = (eachindex(t_plot), latexify(Union{Float64, Int}[0.1, 0.5, 1.5, 5, 10, 50])),
        yticks = latexticks(0:2:10), yaxisposition = :right)
    ylims!(ax4, (0, 10))
    barplot!(ax4, eachindex(t_plot), maxerror_t, label = L"max $\,$")
    barplot!(ax4, eachindex(t_plot), meanerror_t, label = L"mean $\,$")
    axislegend(ax4, position = :rt)

    logNvec = collect(4:8)
    Nvec = 2 .^ logNvec
    maxerror = fill(Inf, length(Nvec))
    meanerror = fill(Inf, length(Nvec))
    delta_x = fill(Inf, length(Nvec))
    t_end = years2seconds(t_plot[end])
    for n in eachindex(Nvec)
        Ni = Nvec[n]
        fname = "nx=$(Ni)_ny=$(Ni)_cpu_interactive_sl=false-dense"
        @load "../data/test1/$fname.jld2" fip
        Omega, c, p, t_out = fip.Omega, fip.c, fip.p, fip.out.t

        ii, jj = slice_along_x(Omega)
        x, y = Omega.X[ii, jj], Omega.Y[ii, jj]
        analytic_solution_r(r) = analytic_solution(r, t_end, c, p, H, R)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        u_numeric = fip.out.u[end][ii, jj]
        abs_error = abs.(u_analytic - u_numeric)

        delta_x[n] = Omega.Wx * 1e-3 / N
        maxerror[n] = maximum(abs_error)
        meanerror[n] = mean(abs_error)
    end
    ax5 = Axis(fig[2, 1], title = L"(c) $\,$", xlabel = L"$N = N_{x} = N_{y} $ (1)",
        ylabel = L"Absolute error (m)$\,$", xscale = log2, yscale = log10,
        xticks = (2 .^ (4:8), [L"$2^{%$l}$" for l in 4:8]),
        yticks = (10. .^ (-1:1), [L"$10^{%$l}$" for l in -1:1]),
        yminorticks = IntervalsBetween(9),
        yminorticksvisible = true,
        yminorgridvisible = true,
        titlegap = tgap,
    )

    mmax = linregression(logNvec, log10.(maxerror))
    mmean = linregression(logNvec, log10.(meanerror))
    scatterlines!(ax5, Nvec, maxerror, label = L"max $\,$", linewidth = lwidth,
        markersize = msize)
    scatterlines!(ax5, Nvec, meanerror, label = L"mean $\,$", linewidth = lwidth,
        markersize = msize)
    lines!(ax5, Nvec, 10 .^ (mmax[1] .* logNvec .+ mmax[2]), linewidth = lwidth,
        linestyle = :dash, color = :gray70 )
    lines!(ax5, Nvec, 10 .^ (mmean[1] .* logNvec .+ mmean[2]), linewidth = lwidth,
        linestyle = :dash, color = :gray30 )
    axislegend(ax5, position = :lb, width = 320,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
    ylims!(ax5, (1e-1, 1e2))

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
    kernels = ["cpu", "gpu"]
    runtime = [zeros(length(Nvec)) for _ in kernels]
    delta_x = [zeros(length(Nvec)) for _ in kernels]
    for k in eachindex(kernels)
        kernel = kernels[k]
        for j in eachindex(Nvec)
            N = Nvec[j]
            fname = "nx=$(N)_ny=$(N)_$(kernel)_interactive_sl=false-dense"
            @load "../data/test1/$fname.jld2" fip

            runtime[k][j] = fip.out.computation_time
            delta_x[k][j] = 2*fip.Omega.Wx * 1e-3 / N
        end
        scatterlines!(ax6, Nvec, runtime[k], label = L"%$kernel $\,$", linewidth = lwidth, markersize = msize)
    end
    mcpu = linregression(logNvec, log10.(runtime[1]))
    mgpu = linregression(logNvec, log10.(runtime[2]))
    lines!(ax6, Nvec, 10 .^ (mcpu[1] .* logNvec .+ mcpu[2]), linewidth = lwidth,
        linestyle = :dash, color = :gray70 )
    lines!(ax6, Nvec, 10 .^ (mgpu[1] .* logNvec .+ mgpu[2]), linewidth = lwidth,
        linestyle = :dash, color = :gray30 )

    axislegend(ax6, position = :lt, width = 200,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)

    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)

    save("plots/test1/finalplot_v0.3.png", fig)
    save("plots/test1/finalplot_v0.3.pdf", fig)

    return maxerror_t, meanerror_t, maxerror, meanerror, runtime, delta_x, mmax, mmean, mcpu, mgpu
end

maxerror_t, meanerror_t, maxerror, meanerror, runtime, delta_x, mmax, mmean, mcpu, mgpu = main()