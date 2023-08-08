
function standard_config(; n::Int = 6)
    T = Float64
    W = T(3000e3)
    Omega = ComputationDomain(W, n)
    return T, Omega
end

function comparison_figure(n)
    fig = Figure()
    axs = [Axis(fig[i, 1]) for i in 1:n]
    return fig, axs
end

function update_compfig!(axs::Vector{Axis}, fi::Vector, bm::Vector)
    if length(axs) == length(fi) == length(bm)
        nothing
    else
        error("Vectors don't have matching length.")
    end

    clr = RGBf(rand(3)...)
    for i in eachindex(axs)
        lines!(axs[i], bm[i], color = clr)
        lines!(axs[i], fi[i], color = clr, linestyle = :dash)
    end
end

function benchmark1()
    # Generating numerical results
    T, Omega = standard_config()
    c = PhysicalConstants(rho_litho = 0.0)
    p = LateralVariability(Omega)

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver = BS3(),
        interactive_geostate = false, verbose = false)
    Omega, p = copystructs2cpu(Omega, p)

    # Comparing to analytical results
    slicex, slicey = slice_along_x(Omega)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    fig, axs = comparison_figure(1)
    x, y = Omega.X[slicex, slicey], Omega.Y[slicex, slicey]
    for k in eachindex(t_out)
        t = t_out[k]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( get_r.(x, y) )
        @test mean(abs.(results.viscous[k][slicex, slicey] .- u_analytic)) < 5
        @test maximum(abs.(results.viscous[k][slicex, slicey] .- u_analytic)) < 12
        update_compfig!(axs, [results.viscous[k][slicex, slicey]], [u_analytic])
    end
    save("plots/benchmark1/plot.png", fig)
end

function benchmark2()
    # Generating numerical results
    T, Omega = standard_config()
    c = PhysicalConstants(rho_ice = 0.931e3)
    G, nu = 0.50605e11, 0.28        # shear modulus (Pa) and Poisson ratio of lithsphere
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]
    p = LateralVariability( Omega, layer_boundaries = lb,
        layer_viscosities = [1e21, 1e21, 2e21], litho_youngmodulus = E,
        litho_poissonratio = nu )
    t_out = years2seconds.([0.0, 1e3, 2e3, 5e3, 1e4, 1e5])
    sl0 = fill(-Inf, Omega.Nx, Omega.Ny)
    slicex, slicey = slice_along_x(Omega)
    x = Omega.X[slicex, slicey]
    data = load_spada()

    for case in ["disc", "cap"]
        # Generate FastIsostasy results
        if occursin("disc", case)
            alpha = T(10)                       # max latitude (°) of uniform ice disc
            Hmax = T(1000)                      # uniform ice thickness (m)
            R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
            H_ice = stereo_ice_cylinder(Omega, R, Hmax)
        elseif occursin("cap", case)
            alpha = T(10)                       # max latitude (°) of ice cap
            Hmax = T(1500)
            H_ice = stereo_ice_cap(Omega, alpha, Hmax)
        end
        results = fastisostasy(t_out, Omega, c, p, H_ice, sealevel_0 = sl0,
            ODEsolver = BS3(), interactive_geostate = true, verbose = false)
        
        # Compare to 1D GIA models benchmark
        fig, axs = comparison_figure(3)
        u_itp_0 = interpolate_spada_benchmark(c, data["u_$case"][1])

        for k in eachindex(t_out)
            u_itp = interpolate_spada_benchmark(c, data["u_$case"][k])
            dudt_itp = interpolate_spada_benchmark(c, data["dudt_$case"][k])
            n_itp = interpolate_spada_benchmark(c, data["n_$case"][k])

            u_bm = u_itp.(x) .- u_itp_0.(x)
            dudt_bm = dudt_itp.(x)
            n_bm = n_itp.(x)

            u_fi = results.viscous[k][slicex, slicey]
            dudt_fi = m_per_sec2mm_per_yr.(results.displacement_rate[k][slicex, slicey])
            n_fi = results.geoid[k][slicex, slicey]

            update_compfig!(axs, [u_fi, dudt_fi, n_fi], [u_bm, dudt_bm, n_bm])

            @test mean(abs.(u_fi .- u_bm)) < 20
            @test mean(abs.(dudt_fi .- dudt_bm)) < 3
            @test mean(abs.(n_fi .- n_bm)) < 3
            # println("$m_u,  $m_dudt, $m_n")
        end
        save("plots/benchmark2/$case.png", fig)
    end
end

function benchmark3()
    T, Omega = standard_config()
    c = PhysicalConstants()
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.( vcat(0:1_000:5_000, 10_000:5_000:50_000) )

    slicex, slicey = slice_along_x(Omega)
    x = Omega.X[slicex, slicey]
    cases = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η",
        "no_litho", "ref"]
    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3", "E0L4V4", "E0L4V4"]
    mean_tol = [10, 10, 30, 30, Inf, 30]
    max_tol = [20, 20, 60, 60, Inf, 60]
    for m in eachindex(cases)
        fig, axs = comparison_figure(1)
        case = cases[m]
        file = seakon_files[m]
        x_sk, u_sk = load_latychev("../data/Latychev/$file", -1.0, 3e3)
        x_sk .*= 1e3

        p = choose_case(case, Omega, c)
        results = fastisostasy(t_out, Omega, c, p, Hcylinder, interactive_geostate = false,
            ODEsolver = BS3(), verbose = false)

        for k in eachindex(t_out)
            itp = linear_interpolation(x_sk, u_sk[:, k], extrapolation_bc = Flat())
            u_fi, u_bm = results.viscous[k][slicex, slicey], itp.(x)
            update_compfig!(axs, [u_fi], [u_bm])
            @test mean(abs.(u_fi .- u_bm)) .< mean_tol[m]
            @test maximum(abs.(u_fi .- u_bm)) .< max_tol[m]
        end
        save("plots/benchmark3/$case.png", fig)
    end
end


function benchmark4()
end

function benchmark5()
end

function benchmark6()
end