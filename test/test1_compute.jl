push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_compute.jl")
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

    timespan = T.([0, 5e4]) * T(c.seconds_per_year)     # (yr) -> (s)
    dt_out = T(100) * T(c.seconds_per_year)             # (yr) -> (s), time step for saving output
    t_vec = timespan[1]:dt_out:timespan[2]              # (s)
    refine = fill(300.0, length(t_vec)-1)

    u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)

    # analytic_support = collect(10.0 .^ (-14:5))

    @testset "analytic solution" begin
        sol = analytic_solution(T(0), T(50000 * c.seconds_per_year), c, p, H, R, analytic_support)
        @test isapprox( sol, -1000*c.ice_density/mean(p.mantle_density), rtol=T(1e-2) )
    end

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    tools = precompute_terms(dt_out, Omega, p, c)

    @time forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz_disc, tools, p, c, dt_refine = refine)
    jldsave(
        "data/test1_$(case)_N=$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        Omega = Omega,
        c = c,
        p = p,
        R = R,
        H = H,
    )

    ##############

    # Computing analytical solution is quite expensive as it involves
    # integration over κ ∈ [0, ∞) --> load precomputed interpolator.
    compute_analytical_sol = false
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
    - "cn_2layers"
    - "cn_3layers"
    - "euler_2layers"
    - "euler_3layers"
"""
case = "euler_2layers"
for n in 7:7
    main(n, case)
end