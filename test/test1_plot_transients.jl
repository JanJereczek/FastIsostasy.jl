push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_plots.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    make_plot = true,
    make_anim = false,
)

    T = Float64
    timespan = years2seconds([0, 5e4])          # (yr) -> (s)
    dt_out = years2seconds(100)                 # (yr) -> (s), time step for saving output
    t_vec = timespan[1]:dt_out:timespan[2]      # (s)

    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    sol = load("data/$(case)_N=$(Omega.N).jld2")


    # Computing analytical solution is quite expensive as it involves
    # integration over κ ∈ [0, ∞) --> load precomputed interpolator.
    compute_analytical_sol = true
    tend = T(Inf * c.seconds_per_year)

    if compute_analytical_sol
        analytic_solution_r(r) = analytic_solution(r, tend, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( sqrt.(Omega.X .^ 2 + Omega.Y .^ 2) )
        U = [u_analytic[i,j] for i in axes(Omega.X, 1), j in axes(Omega.X, 2)]
        u_analytic_interp = linear_interpolation(
            (Omega.X[1,:], Omega.Y[:,1]),
            u_analytic,
            extrapolation_bc = NaN,
        )
        jldsave(
            "data/analytical_solution_interpolator_N=$(Omega.N).jld2",
            u_analytic_interp = u_analytic_interp,
        )
    else
        u_analytic_interp = load(
            "data/analytical_solution_interpolator_N=$(Omega.N).jld2",
            "u_analytic_interp",
        )
    end
    u_analytic = u_analytic_interp.(Omega.X, Omega.Y)

    u_plot = [
        u_analytic,
        u3D_viscous[:,:,end],
        u3D_elastic[:,:,end],
        u3D_viscous[:,:,end] - u_analytic,
        u3D_elastic[:,:,end] + u3D_viscous[:,:,end],
    ]
    panels = [
        (2,1),
        (1,2),
        (1,3),
        (2,2),
        (2,3),
    ]
    labels = [
        L"Analytical solution for viscous displacement (m) $\,$",
        L"Vertical displacement of viscous response $u^V$ (m)",
        L"Vertical displacement of elastic response $u^E$ (m)",
        L"Numerical minus analytical solution $u^V - u^A$ (m)",
        L"Total vertical displacement $u^E + u^V$ (m)",
    ]
    if make_plot
        response_fig = plot_response(
            Omega,
            sigma_zz_disc,
            u_plot,
            panels,
            labels,
            case,
        )
    end

    if make_anim
        anim_name = "plots/discload_$(case)_N=$(Omega.N)"
        animate_viscous_response(t_vec, Omega, u3D_viscous, anim_name, (-300.0, 50.0))
    end
end

"""
Application cases:
    - "homegeneous_viscosity_2layers"
    - "homogeneous_viscosity_3layers"
    - "euler_2layers"
    - "euler_3layers"
"""
case = "euler_2layers"
for n in 8:8
    main(n, case)
end