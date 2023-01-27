push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using SpecialFunctions
using JLD2
using LinearAlgebra
include("helpers_compute.jl")
include("helpers_plot.jl")

@inline function main()

    case = "euler3layers"
    Nvec = 2 .^ (4:8)
    fig = Figure(resolution = (1600, 900))

    maxerror = Float64[]
    meanerror = Float64[]
    delta_x = Float64[]
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    t_end = years2seconds(1e5)

    for N in Nvec
        hash = "$(case)_gpu_N$(N)"
        sol = load("data/test1/$hash.jld2")

        islice, jslice = Int(round(N/2)), Int(round(N/2))
        Omega, c, p, H, R = sol["Omega"], sol["c"], sol["p"], sol["H"], sol["R"]
        x, y = Omega.X[islice, jslice:end], Omega.Y[islice, jslice:end] 
        analytic_solution_r(r) = analytic_solution(r, t_end, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        u_numeric = sol["u3D_viscous"][:, :, end]
        abs_error = abs.(u_analytic - u_numeric[islice, jslice:end])

        append!(delta_x, Omega.L * 1e-3 / N)
        append!(maxerror, maximum(abs_error))
        append!(meanerror, mean(abs_error))
    end
    ax1 = Axis(
        fig[1,1],
        xlabel = L"$\Delta x $ (km)",
        ylabel = L"Error w.r.t. analytical solution $\,$",
        yscale = log10,
    )
    scatterlines!(ax1, delta_x, maxerror, label = L"Maximum $\,$")
    scatterlines!(ax1, delta_x, meanerror, label = L"Average $\,$")
    axislegend(ax1, position = :rb)


    ax2 = Axis(
        fig[1,2],
        xlabel = L"$\Delta x $ (km)",
        ylabel = L"Run time (s) $\,$",
        yscale = log10,
    )
    for kernel in ["cpu", "gpu"]
        runtime = Float64[]
        delta_x = Float64[]
        for N in Nvec
            hash = "$(case)_$(kernel)_N$(N)"
            sol = load("data/test1/$hash.jld2")
            append!(runtime, sol["t_fastiso"])
            append!(delta_x, sol["Omega"].L * 1e-3 / N)
        end
        scatterlines!(ax2, delta_x, runtime, label = L"%$kernel $\,$")
    end
    axislegend(ax2, position = :rt)


    N = 256
    N2 = Int(N/2)
    N4 = Int(N/4)
    hash = "$(case)_gpu_N$(N)"
    sol = load("data/test1/$hash.jld2")
    Omega, c, p, H, R = sol["Omega"], sol["c"], sol["p"], sol["H"], sol["R"]
    t_out = sol["t_out"]

    t_plot = years2seconds.([100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    colors = [:black, :orange, :blue, :red, :gray, :purple]
    x, y = diag(Omega.X)[N2:N2+N4], diag(Omega.Y)[N2:N2+N4]
    ax3 = Axis(fig[1, 3], xlabel = L"$x$ (m)", ylabel = L"$u^v$ (m)")
    for i in eachindex(t_plot)
        t = t_plot[i]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
        k = argmin( (t_out .- t) .^ 2 )
        u_numeric = diag(sol["u3D_viscous"][:, :, k])
        tyr = Int(round(seconds2years(t)))

        if i == 1
            lines!(
                ax3,
                x,
                u_numeric[N2:N2+N4],
                label = L"numeric $\,$",
                color = colors[i],
            )
            lines!(
                ax3,
                x,
                u_analytic,
                label = L"analytic $\,$",
                linestyle = :dash,
                color = colors[i],
            )
        else
            lines!(
                ax3,
                x,
                u_numeric[N2:N2+N4],
                color = colors[i],
            )
            lines!(
                ax3,
                x,
                u_analytic,
                linestyle = :dash,
                color = colors[i],
            )
        end
        text!(
            ax3,
            x[1],
            u_numeric[N2] + 10,
            text = "$tyr yr",
            align = (:left, :bottom),
            color = colors[i],
        )
    end
    axislegend(ax3, position = :rb)

    t_2Dplot = [100.0, 1000.0, 100000.0]
    clim = (-300, 10)
    cmap = cgrad(:PuOr_5, rev = true)

    for i in eachindex(t_2Dplot)
        tyr = Int(round(t_2Dplot[i]))
        t = years2seconds(t_2Dplot[i])
        k = argmin( (t_out .- t) .^ 2 )
        u_numeric = sol["u3D_viscous"][:, :, k]

        ax3D = Axis3(fig[2, i], title = L"$t = %$tyr$ yr")

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
            linewidth = 0.1,
            color = :black,
        )
    end
    Colorbar(
        fig[3, :],
        colorrange = clim,
        colormap = cmap,
        label = L"Viscous displacement $u^V$ (m)$",
        vertical = false,
        width = Relative(0.5),
    )

    save("plots/test1/finalplot.png", fig)
    save("plots/test1/finalplot.pdf", fig)
end

main()