push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_plot.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    make_plot = true,
    make_anim = false,
)

    T = Float64
    L = T(2000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)

    if occursin("2layers", case)
        eta_channel = fill(1e21, size(Omega.X)...)
        p = init_solidearth_params(T, Omega, channel_viscosity = eta_channel)
    elseif occursin("3layers", case)
        p = init_solidearth_params(T, Omega)
    end
    c = init_physical_constants(T)

    timespan = years2seconds.([0.0, 5e4])       # (yr) -> (s)
    dt_out = years2seconds(100.0)               # (yr) -> (s), time step for saving output
    t_vec = timespan[1]:dt_out:timespan[2]      # (s)
    sol = load("data/$(case)_N=$(Omega.N).jld2")

    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    # analytic_support = collect(10.0 .^ (-14:.5:0))

    t_plot = years2seconds.([100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    colors = [:black, :orange, :blue, :red, :gray, :purple]
    islice, jslice = Int(round(Omega.N/2)), Int(round(Omega.N/2))
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

        t_idx = Int(round(t / dt_out)) + 1
        u_numeric = sol["u3D_viscous"][islice:end, jslice, t_idx]

        lines!(x, u_numeric, label = i == 1 ? L"numeric $\,$" : " ", color = colors[i])
        lines!(x, u_analytic, label = i == 1 ? L"analytic $\,$" : " ", linestyle = :dash, color = colors[i])
    end

    save("plots/test1_transients_N$(Omega.N).png", fig)
    save("plots/test1_transients_N$(Omega.N).pdf", fig)
end

"""
Application cases:
    - "homegeneous_viscosity_2layers"
    - "homogeneous_viscosity_3layers"
    - "euler_2layers"
    - "euler_3layers"
"""
case = "euler_2layers"
for n in 6:6
    main(n, case)
end