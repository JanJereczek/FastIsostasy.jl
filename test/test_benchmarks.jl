function benchmark1_compare(domain::RegionalDomain, sim, H, R)
    # Comparing to analytical results
    ii, jj = slice_along_x(domain)
    fig, axs = comparison_figure(1)
    x, y = domain.X[ii, jj], domain.Y[ii, jj]
    cmap = cgrad(:jet, length(sim.out.t), categorical = true)
    
    for k in eachindex(sim.out.t)[2:end]
        t = years2seconds(sim.out.t[k])
        analytic_solution_r(r) = analytic_solution(r, t, sim.c, sim.p, H, R)
        u_analytic = analytic_solution_r.( get_r.(x, y) )

        mean_error = mean(abs.(sim.out.u[k][ii, jj] .- u_analytic))
        max_error = maximum(abs.(sim.out.u[k][ii, jj] .- u_analytic))

        # @show mean_error, max_error
        @test mean_error < 6
        @test max_error < 7

        update_compfig!(axs, [sim.out.u[k][ii, jj]], [u_analytic], cmap[k])
    end
    return fig
end

function benchmark1()
    # Generating numerical results
    T = Float64
    use_cuda = false
    sim = benchmark1_sim(T, use_cuda)
    run!(sim)
    println("Computation took $(sim.nout.computation_time) s")
end

function benchmark1_float32()
    T = Float32
    use_cuda = false
    sim = benchmark1_sim(T, use_cuda)
    run!(sim)
    println("Computation took $(sim.nout.computation_time) s")
    return nothing
end


function benchmark1_external_loadupdate()
    T = Float32
    use_cuda = false
    sim = benchmark1_sim(T, use_cuda)
    integrator = init_integrator(sim)

    for k in eachindex(sim.nout.t)[2:end]
        step!(integrator, sim.nout.t[k] - sim.nout.t[k-1], true)
    end
end

function benchmark1_gpu()
    T = Float32
    use_cuda = true
    sim = benchmark1_sim(T, use_cuda)
    run!(sim)
    println("Computation took $(sim.nout.computation_time) s")
end

function benchmark2()
    T = Float32
    domain = RegionalDomain(3000f3, 7, correct_distortion = false)
    c = PhysicalConstants(rho_ice = T(0.931f3))

    G, nu = 0.50605f11, 0.28f0
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301f3, 5951f3, 5701f3]
    p = SolidEarthParameters(domain, layer_boundaries = lb,
        layer_viscosities = [1f21, 1f21, 2f21], litho_youngmodulus = E,
        litho_poissonratio = nu, rho_litho = 2.8f3)

    εt = 1f-8
    t_out = [-εt, 0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5]
    t_Hice = [-εt, 0.0, t_out[end]]
    b = fill(1e6, domain.nx, domain.ny)
    ii, jj = slice_along_x(domain)
    theta = rad2deg.(domain.Theta[ii, jj])
    ii = ii[theta .< 20]
    theta = rad2deg.(domain.Theta[ii, jj])
    (_, _), X, Xitp = load_spada2011()
    opts = SolverOptions(interactive_sealevel = true, verbose = true,
        diffeq = DiffEqOptions(reltol = 1e-4))

    for case in ["disc", "cap"]
        # Generate FastIsostasy results
        if occursin("disc", case)
            alpha = 10.0                        # max latitude (°) of uniform ice disc
            Hmax = 1000.0                       # uniform ice thickness (m)
            R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
            H_ice = stereo_ice_cylinder(domain, R, Hmax)
        elseif occursin("cap", case)
            alpha = 10.0                        # max latitude (°) of ice cap
            Hmax = 1500.0
            H_ice = stereo_ice_cap(domain, alpha, Hmax)
        end
        Hice = [zeros(domain.nx, domain.ny), H_ice, H_ice]
        mask = collect(H_ice .> 1e-8)
        sim = Simulation(domain, c, p, t_out, t_Hice, Hice, opts = opts, b_0 = b,
            maskactive = mask, output = "intermediate")
        run!(sim)
        
        # Compare to 1D GIA models benchmark
        fig, axs = comparison_figure(3)
        u_0 = Xitp["u_$case"].(theta, 0)
        cmap = cgrad(:jet, length(sim.out.t), categorical = true)

        for k in eachindex(t_out)[3:end]
            tt = t_out[k]
            u_bm = Xitp["u_$case"].(theta, tt) .- u_0
            dudt_bm = Xitp["dudt_$case"].(theta, tt)
            n_bm = Xitp["n_$case"].(theta, tt)

            u_fi = sim.out.u[k][ii, jj]
            dudt_fi = sim.out.dudt[k][ii, jj] .* 1e3    # convert m/yr to mm/yr
            dz_ss_fi = sim.out.dz_ss[k][ii, jj]

            update_compfig!(axs, [u_fi, dudt_fi, dz_ss_fi], [u_bm, dudt_bm, n_bm], cmap[k])

            m_u = mean(abs.(u_fi .- u_bm))
            m_dudt = mean(abs.(dudt_fi .- dudt_bm))
            m_n = mean(abs.(dz_ss_fi .- n_bm))
            println("$m_u, $m_dudt, $m_n")
            @test m_u < 27
            @test m_dudt < 8
            @test m_n < 4
        end
        if SAVE_PLOTS
            isdir("plots/benchmark2") || mkdir("plots/benchmark2")
            save("plots/benchmark2/$case.png", fig)
        end
    end
end

function benchmark3()
    domain = RegionalDomain(3000e3, 6)
    c = PhysicalConstants()

    R = 1000e3                  # ice disc radius (m)
    H = 1e3                     # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(domain, R, H)
    Hice = [zeros(domain.nx, domain.ny), Hcylinder, Hcylinder]

    t_out = [0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0]
    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]

    ii, jj = slice_along_x(domain)
    x = domain.X[ii, jj]
    cmap = cgrad(:jet, length(t_out), categorical = true)

    cases = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η",
        "no_litho", "ref"]
    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3", "E0L0V1", "E0L4V4"]
    mean_tol = [7, 8, 15, 11, 8, 15]
    max_tol = [13, 18, 28, 22, 23, 42]

    for m in eachindex(cases)
        fig, axs = comparison_figure(1)
        case = cases[m]
        file = seakon_files[m]
        _, _, usk_itp = load_latychev_test3(case = file)
        p, _, _ = choose_case(case, domain)
        opts = SolverOptions(diffeq = DiffEqOptions(reltol = 1e-5), verbose = true)

        sim = Simulation(domain, c, p, t_out, t_Hice, Hice, opts = opts, output = "sparse")
        run!(sim)

        println("---------------")
        println("Case: $case")
        for k in eachindex(t_out)[2:end]
            u_bm = usk_itp.(x ./ 1e3, t_out[k])
            u_fi = sim.out.u[k][ii, jj] + sim.out.ue[k][ii, jj]
            update_compfig!(axs, [u_fi], [u_bm], cmap[k])
            emean = mean(abs.(u_fi .- u_bm))
            emax = maximum(abs.(u_fi .- u_bm))
            @show emean, emax, mean_tol[m], max_tol[m]
            @test emean .< mean_tol[m]
            @test emax .< max_tol[m]
        end
        if SAVE_PLOTS
            isdir("plots/benchmark3") || mkdir("plots/benchmark3")
            save("plots/benchmark3/$case.png", fig)
        end
    end
end


function benchmark4()
end
