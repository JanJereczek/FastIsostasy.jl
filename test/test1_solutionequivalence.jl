push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using JLD2
using CairoMakie
using LinearAlgebra
include("helpers_plot.jl")

function lineplot_timesteps!(ax1, sol, N, t_plot, linestyle)
    Omega, p, c, t_out, r, N2 = extract_data_from_solution(sol, N)
    for t in t_plot
        lineplot_timestep!(ax1, t, t_out, r, sol, linestyle)
    end
end

function extract_data_from_solution(sol, N)
    Omega, p, c, t_out = sol["Omega"], sol["p"], sol["c"], sol["t_out"]
    r = sqrt.( diag(Omega.X) .^ 2 + diag(Omega.Y) .^ 2 )
    N2 = Int(N/2)
    r = vcat( -r[ 1:N2 ], r[N2+1:end] )
    return Omega, p, c, t_out, r, N2
end

function lineplot_timestep!(ax1, t, t_out, r, sol, linestyle)
    k = argmin( (t_out .- t) .^ 2 )
    u = diag(sol["u3D_viscous"][:, :, k])
    lines!(ax1, r, u, linestyle = linestyle)
end

@inline function main(
    n::Int,                     # 2^n x 2^n cells on domain, (1)
)

    N = 2^n
    kernels = ["cpu","gpu"]
    methods = ["cn", "euler"]
    layers = ["2layers", "3layers"]
    linestyles = [:solid, :dash]

    fig = Figure(resolution=(1600, 700))
    ax1 = Axis(
        fig[1, 1],
        xlabel = L"Distance to pole $r$ (m)",
        ylabel = L"Viscous displacement $u^v$ (m)",
    )
    ax2 = Axis(
        fig[1, 2],
        xlabel = L"Distance to pole $r$ (m)",
        yticklabelsvisible = false,
    )
    ax3 = Axis(
        fig[1, 3],
        xlabel = L"Distance to pole $r$ (m)",
        yticklabelsvisible = false,
    )

    t_plot = years2seconds.([100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    for (kernel, linestyle) in zip(kernels, linestyles)
        case = "euler2layers"
        sol = load("data/test1/$(case)_$(kernel)_N$(N).jld2")
        lineplot_timesteps!(ax1, sol, N, t_plot, linestyle)
    end

    for (method, linestyle) in zip(methods, linestyles)
        case = string(method, "2layers")
        kernel = "gpu"
        sol = load("data/test1/$(case)_$(kernel)_N$(N).jld2")
        lineplot_timesteps!(ax2, sol, N, t_plot, linestyle)
    end

    for (layer, linestyle) in zip(layers, linestyles)
        case = string("euler", layer)
        kernel = "gpu"
        sol = load("data/test1/$(case)_$(kernel)_N$(N).jld2")
        lineplot_timesteps!(ax3, sol, N, t_plot, linestyle)
    end

    save("plots/test1/equivalence_N$(N).png", fig)
    save("plots/test1/equivalence_N$(N).pdf", fig)
end

for n in 6:8
    main(n)
end