push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_plot.jl")
include("helpers_compute.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    make_anim = false,
)

    N = 2^n
    sol = load("data/test1/$(case)_N$(N).jld2")
    R, H, Omega, c, p = sol["R"], sol["H"], sol["Omega"], sol["c"], sol["p"]
    t_out = sol["t_out"]
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:0))

    t_plot = years2seconds.([100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    colors = [:black, :orange, :blue, :red, :gray, :purple]
    islice, jslice = Int(round(N/2)), Int(round(N/2))
    x = Omega.X[islice, jslice:end]

    fig = Figure(resolution = (1000, 800))
    ax = Axis(fig[1, 1], xlabel = L"$x$ (m)", ylabel = L"$u^v$ (m)")
    for i in eachindex(t_plot)
        t = t_plot[i]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.(
            Omega.X[islice, jslice:end] .^ 2 +
            Omega.Y[islice, jslice:end] .^ 2 
        ) )

        k = argmin( (t_out .- t).^2 )
        u_numeric = sol["u3D_viscous"][islice:end, jslice, k]

        lines!(
            ax,
            x,
            u_numeric,
            label = i == 1 ? L"numeric $\,$" : " ",
            color = colors[i],
        )
        lines!(
            ax,
            x,
            u_analytic,
            label = i == 1 ? L"analytic $\,$" : " ",
            linestyle = :dash,
            color = colors[i],
        )
    end

    save("plots/test1/$(case)_transients_N$N.png", fig)
    save("plots/test1/$(case)_transients_N$N.pdf", fig)
end

"""
Application cases:
    - "cn2layers"
    - "cn3layers"
    - "euler2layers_cpu"
    - "euler3layers_cpu"
    - "euler2layers_gpu"
    - "euler3layers_gpu"
"""
case = "euler3layers_cpu"
for n in 4:6
    main(n, case)
end