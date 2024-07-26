function benchmark1_compare(Omega, fip, H, R)
    # Comparing to analytical results
    ii, jj = slice_along_x(Omega)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    fig, axs = comparison_figure(1)
    x, y = Omega.X[ii, jj], Omega.Y[ii, jj]
    cmap = cgrad(:jet, length(fip.out.t), categorical = true)
    
    for k in eachindex(fip.out.t)[2:end]
        t = years2seconds(fip.out.t[k])
        analytic_solution_r(r) = analytic_solution(r, t, fip.c, fip.p, H, R)
        u_analytic = analytic_solution_r.( get_r.(x, y) )

        mean_error = mean(abs.(fip.out.u[k][ii, jj] .- u_analytic))
        max_error = maximum(abs.(fip.out.u[k][ii, jj] .- u_analytic))
        # @show mean_error, max_error

        @test mean_error < 6
        @test max_error < 7

        update_compfig!(axs, [fip.out.u[k][ii, jj]], [u_analytic], cmap[k])
    end
    return fig
end

function benchmark1()
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 7, correct_distortion = false)
    c, p, t_out, R, H, t_Hice, Hice = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, output = "sparse")
    solve!(fip)
    # println("Computation took $(fip.out.computation_time) s")
    fig = benchmark1_compare(Omega, fip, H, R)
    if SAVE_PLOTS
        save("plots/benchmark1/plot.png", fig)
    end
end

function benchmark1_gpu()
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 7, use_cuda = true, correct_distortion = false)
    c, p, t_out, R, H, t_Hice, Hice = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, output = "sparse")
    solve!(fip)
    # println("Computation took $(fip.out.computation_time) s")
    Omega, p = reinit_structs_cpu(Omega, p)

    fig = benchmark1_compare(Omega, fip, H, R)
    if SAVE_PLOTS
        save("plots/benchmark1/plot_gpu.png", fig)
    end
end

function benchmark1_external_loadupdate()
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 7, correct_distortion = false)
    c, p, t_out, R, H, t_Hice, Hice = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, output = "sparse")
    update_diagnostics!(fip.now.dudt, fip.now.u, fip, 0.0)
    write_out!(fip.out, fip.now, 1)
    ode = init(fip)
    @inbounds for k in eachindex(fip.out.t)[2:end]
        step!(fip, ode, (fip.out.t[k-1], fip.out.t[k]))
        write_out!(fip.out, fip.now, k)
    end
    # println("Computation took $(fip.out.computation_time) s")

    fig = benchmark1_compare(Omega, fip, H, R)
    if SAVE_PLOTS
        save("plots/benchmark1/plot_external_loadupdate.png", fig)
    end
end

function benchmark2()
    Omega = ComputationDomain(3000e3, 6)
    c = PhysicalConstants(rho_ice = 0.931e3, rho_litho = 2.8e3)

    G, nu = 0.50605e11, 0.28        # shear modulus (Pa) and Poisson ratio of lithsphere
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]
    p = LayeredEarth( Omega, layer_boundaries = lb,
        layer_viscosities = [1e21, 1e21, 2e21], litho_youngmodulus = E,
        litho_poissonratio = nu )

    εt = 1e-8
    t_out = [-εt, 0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5]
    t_Hice = [-εt, 0.0, t_out[end]]
    b = fill(1e6, Omega.Nx, Omega.Ny)
    ii, jj = slice_along_x(Omega)
    theta = rad2deg.(Omega.Theta[ii, jj])
    ii = ii[theta .< 20]
    theta = rad2deg.(Omega.Theta[ii, jj])
    (_, _), X, Xitp = load_spada2011()
    opts = SolverOptions(interactive_sealevel = true, verbose = true,
        diffeq = (alg=Tsit5(), reltol=1e-4))

    for case in ["disc", "cap"]
        # Generate FastIsostasy results
        if occursin("disc", case)
            alpha = 10.0                        # max latitude (°) of uniform ice disc
            Hmax = 1000.0                       # uniform ice thickness (m)
            R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
            H_ice = stereo_ice_cylinder(Omega, R, Hmax)
        elseif occursin("cap", case)
            alpha = 10.0                        # max latitude (°) of ice cap
            Hmax = 1500.0
            H_ice = stereo_ice_cap(Omega, alpha, Hmax)
        end
        Hice = [zeros(Omega.Nx, Omega.Ny), H_ice, H_ice]
        mask = collect(H_ice .> 1e-8)
        fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts, b_0 = b,
            maskactive = mask, output = "intermediate")
        solve!(fip)
        
        # Compare to 1D GIA models benchmark
        fig, axs = comparison_figure(3)
        u_0 = Xitp["u_$case"].(theta, 0)
        cmap = cgrad(:jet, length(fip.out.t), categorical = true)

        for k in eachindex(t_out)[3:end]
            tt = t_out[k]
            u_bm = Xitp["u_$case"].(theta, tt) .- u_0
            dudt_bm = Xitp["dudt_$case"].(theta, tt)
            n_bm = Xitp["n_$case"].(theta, tt)

            u_fi = fip.out.u[k][ii, jj]
            dudt_fi = fip.out.dudt[k][ii, jj] .* 1e3    # convert m/yr to mm/yr
            dz_ss_fi = fip.out.dz_ss[k][ii, jj]

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
            save("plots/benchmark2/$case.png", fig)
        end
    end
end

function benchmark3()
    Omega = ComputationDomain(3000e3, 6)
    c = PhysicalConstants()

    R = 1000e3                  # ice disc radius (m)
    H = 1e3                     # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]

    t_out = [0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0]
    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]

    ii, jj = slice_along_x(Omega)
    x = Omega.X[ii, jj]
    cmap = cgrad(:jet, length(t_out), categorical = true)

    cases = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η",
        "no_litho", "ref"]
    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3", "E0L0V1", "E0L4V4"]
    mean_tol = [10, 10, 15, 10, 10, 15]
    max_tol = [15, 20, 30, 25, 25, 45]

    for m in eachindex(cases)
        fig, axs = comparison_figure(1)
        case = cases[m]
        file = seakon_files[m]
        _, _, usk_itp = load_latychev_test3(case = file)
        p, _, _ = choose_case(case, Omega)
        tol = occursin("_D", case) ? 1e-5 : 1e-4
        opts = SolverOptions(diffeq = (alg = Tsit5(), reltol = tol), verbose = true)

        fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts, output = "sparse")
        solve!(fip)

        println("---------------")
        for k in eachindex(t_out)[2:end]
            u_bm = usk_itp.(x ./ 1e3, t_out[k])
            u_fi = fip.out.u[k][ii, jj] + fip.out.ue[k][ii, jj]
            update_compfig!(axs, [u_fi], [u_bm], cmap[k])
            emean = mean(abs.(u_fi .- u_bm))
            emax = maximum(abs.(u_fi .- u_bm))
            println("$emax,  $emean")
            @test emean .< mean_tol[m]
            @test emax .< max_tol[m]
        end
        if SAVE_PLOTS
            save("plots/benchmark3/$case.png", fig)
        end
    end
end


function benchmark4()
end
