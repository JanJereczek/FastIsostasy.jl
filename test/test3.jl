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
    make_plot = true,
    make_anim = false,
)

    T = Float64

    L = T(2000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)

    rmax = sqrt( 2*L^2 )
    eta0 = 1e18
    etamax = 1e21
    delta_eta = (etamax - eta0) / rmax 
    radial_eta(r) = eta0 + delta_eta * r
    eta_r = radial_eta.(get_r.(Omega.X, Omega.Y))

    p = init_solidearth_params(
        T,
        Omega,
        channel_viscosity = eta_r,
        halfspace_viscosity = eta_r,
    )
    c = init_physical_constants(T)

    timespan = T.([0, 5e4]) .* T(c.seconds_per_year)    # (yr) -> (s)
    dt_out = T(100) * T(c.seconds_per_year)             # (yr) -> (s)
    t_out = timespan[1]:dt_out:timespan[2]              # (s)

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    
    domains = 10 .^ (-14:0.05:10)

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    tools = precompute_terms(dt, Omega, p, c)

    @time forward_isostasy!(Omega, t_out, u3D_elastic, u3D_viscous, sigma_zz_disc, tools, p, c)

    jldsave(
        "data/discload_radialeta_N=$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
    )

    ##############

    t_analytic = (10.0 .^ (1:4)) .* T(c.seconds_per_year)
    radial_analytical_solution = zeros(T, size(Omega.X)..., length(t_analytic))
    for k in eachindex(t_analytic)
        for i in axes(Omega.X, 1)
            for j in axes(Omega.X, 2)
                radial_analytical_solution[i, j, k] = analytic_radial_solution(
                    Omega,
                    i,
                    j,
                    t_analytic[k],
                    c,
                    p,
                    H,
                    R,
                    domains,
                )
            end
        end
    end

    u_plot = [
        radial_analytical_solution[:, :, end],
        u3D_viscous[:,:,end],
        u3D_elastic[:,:,end],
        u3D_viscous[:,:,end] - radial_analytical_solution[:, :, end],
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
            "analytical_discload_radialeta",
        )
    end

    if make_anim
        anim_name = "plots/discload_radialeta_N=$(Omega.N)"
        animate_viscous_response(t_vec, Omega, u3D_viscous, anim_name, (-300.0, 50.0))
    end
end

for n in 5:5
    main(n)
end