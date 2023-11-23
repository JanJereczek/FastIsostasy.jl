function comparison_figure(n)
    fig = Figure()
    axs = [Axis(fig[i, 1]) for i in 1:n]
    return fig, axs
end

function update_compfig!(axs::Vector{Axis}, fi::Vector, bm::Vector, clr)
    if length(axs) == length(fi) == length(bm)
        nothing
    else
        error("Vectors don't have matching length.")
    end

    for i in eachindex(axs)
        lines!(axs[i], bm[i], color = clr)
        lines!(axs[i], fi[i], color = clr, linestyle = :dash)
    end
end

function benchmark1_constants(Omega)
    c = PhysicalConstants(rho_litho = 0.0)
    p = LayeredEarth(Omega)
    R, H = 1000e3, 1e3
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    interactive_sealevel = false
    return c, p, R, H, Hcylinder, t_out, interactive_sealevel
end

function benchmark1_compare(Omega, fip, H, R)
    # Comparing to analytical results
    ii, jj = slice_along_x(Omega)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    fig, axs = comparison_figure(1)
    x, y = Omega.X[ii, jj], Omega.Y[ii, jj]
    cmap = cgrad(:jet, length(fip.out.t), categorical = true)
    for k in eachindex(fip.out.t)
        t = fip.out.t[k]
        analytic_solution_r(r) = analytic_solution(r, t, fip.c, fip.p, H, R)
        u_analytic = analytic_solution_r.( get_r.(x, y) )
        @test mean(abs.(fip.out.u[k][ii, jj] .- u_analytic)) < 6
        @test maximum(abs.(fip.out.u[k][ii, jj] .- u_analytic)) < 8
        update_compfig!(axs, [fip.out.u[k][ii, jj]], [u_analytic], cmap[k])
    end
    return fig
end

function benchmark1()
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 7, correct_distortion = false)
    c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)
    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")
    fig = benchmark1_compare(Omega, fip, H, R)
    if SAVE_PLOTS
        save("plots/benchmark1/plot.png", fig)
    end
end

function benchmark1_gpu()
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 7, use_cuda = true, correct_distortion = false)
    c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)
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
    Omega = ComputationDomain(3000e3, 7)
    c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)

    update_diagnostics!(fip.now.dudt, fip.now.u, fip, 0.0)
    write_out!(fip, 1)
    ode = init(fip)
    @inbounds for k in eachindex(fip.out.t)[2:end]
        step!(fip, ode, (fip.out.t[k-1], fip.out.t[k]))
        write_out!(fip, k)
    end
    # println("Computation took $(fip.out.computation_time) s")

    fig = benchmark1_compare(Omega, fip, H, R)
    if SAVE_PLOTS
        save("plots/benchmark1/plot_external_loadupdate.png", fig)
    end
end

function benchmark2()
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 6)
    c = PhysicalConstants(rho_ice = 0.931e3, rho_uppermantle = 3.6e3, rho_litho = 2.7e3)
    G, nu = 0.50605e11, 0.28        # shear modulus (Pa) and Poisson ratio of lithsphere
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]
    p = LayeredEarth( Omega, layer_boundaries = lb,
        layer_viscosities = [1e21, 1e21, 2e21], litho_youngmodulus = E,
        litho_poissonratio = nu )
    t_out = years2seconds.([0.0, 1e3, 2e3, 5e3, 1e4, 1e5])
    sl0 = fill(0.0, Omega.Nx, Omega.Ny)
    ii, jj = slice_along_x(Omega)
    theta = rad2deg.(Omega.Theta[ii, jj])
    ii = ii[theta .< 20]
    theta = rad2deg.(Omega.Theta[ii, jj])
    (_, _), X, Xitp = load_spada2011()

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
        fip = FastIsoProblem(Omega, c, p, t_out, true, H_ice, seasurfaceheight_0 = sl0,
            diffeq = (alg = Tsit5(), reltol = 1e-4))
        solve!(fip)
        
        # Compare to 1D GIA models benchmark
        fig, axs = comparison_figure(3)
        u_0 = Xitp["u_$case"].(theta, 0)
        cmap = cgrad(:jet, length(fip.out.t), categorical = true)

        for k in eachindex(t_out)
            tt = seconds2years(t_out[k])
            u_bm = Xitp["u_$case"].(theta, tt) .- u_0
            dudt_bm = Xitp["dudt_$case"].(theta, tt)
            n_bm = Xitp["n_$case"].(theta, tt)

            u_fi = fip.out.u[k][ii, jj]
            dudt_fi = m_per_sec2mm_per_yr.(fip.out.dudt[k][ii, jj])
            n_fi = fip.out.geoid[k][ii, jj]

            update_compfig!(axs, [u_fi, dudt_fi, n_fi], [u_bm, dudt_bm, n_bm], cmap[k])

            m_u = mean(abs.(u_fi .- u_bm))
            m_dudt = mean(abs.(dudt_fi .- dudt_bm))
            m_n = mean(abs.(n_fi .- n_bm))
            @test m_u < 22
            @test m_dudt < 8
            @test m_n < 4.1
            # println("$m_u,  $m_dudt, $m_n")
        end
        if SAVE_PLOTS
            save("plots/benchmark2/$case.png", fig)
        end
    end
end

function benchmark3()
    Omega = ComputationDomain(3000e3, 6)
    c = PhysicalConstants()
    Hcylinder = uniform_ice_cylinder(Omega, 1000e3, 1e3)
    interactive_sealevel = false
    t_out = years2seconds.( vcat(0:1_000:5_000, 10_000:5_000:50_000) )

    ii, jj = slice_along_x(Omega)
    x = Omega.X[ii, jj]
    cmap = cgrad(:jet, length(t_out), categorical = true)

    cases = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η",
        "no_litho", "ref"]
    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3", "E0L0V1", "E0L4V4"]
    mean_tol = [12, 12, 16, 15, 10, 20]
    max_tol = [24, 30, 30, 35, 20, 45]

    for m in eachindex(cases)
        fig, axs = comparison_figure(1)
        case = cases[m]
        file = seakon_files[m]
        _, _, usk_itp = load_latychev_test3(case = file)

        p, _, _ = choose_case(case, Omega)
        fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)
        solve!(fip)

        # println("---------------")
        for k in eachindex(t_out)
            u_bm = usk_itp.(x ./ 1e3, seconds2years(t_out[k]))
            u_fi = fip.out.u[k][ii, jj] + fip.out.ue[k][ii, jj]
            update_compfig!(axs, [u_fi], [u_bm], cmap[k])
            emean = mean(abs.(u_fi .- u_bm))
            emax = maximum(abs.(u_fi .- u_bm))
            # println("$emax,  $emean")
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

function benchmark5()
    Omega = ComputationDomain(3000e3, 5)
    c = PhysicalConstants()
    lb = [88e3, 100e3, 200e3, 300e3]
    dims, logeta, logeta_itp = load_wiens2022(extrapolation_bc = Flat())
    lv = 10 .^ cat([logeta_itp.(Omega.X, Omega.Y, z) for z in lb]..., dims=3)
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    R, H = 1000e3, 1e3
    Hice = uniform_ice_cylinder(Omega, R, H, center = [-1000e3, -1000e3])
    t_out = years2seconds.(1e3:1e3:2e3)
    fip = FastIsoProblem(Omega, c, p, t_out, false, Hice)
    solve!(fip)
    ground_truth = copy(p.effective_viscosity)

    config = InversionConfig(N_iter = 15)
    data = InversionData(copy(fip.out.t[2:end]), copy(fip.out.u[2:end]), copy([Hice, Hice]),
        config)
    paraminv = InversionProblem(deepcopy(fip), config, data)
    solve!(paraminv)
    logeta, Gx, abserror = extract_inversion(paraminv)

    if SAVE_PLOTS
        p_estim = copy(ground_truth)
        p_estim[paraminv.data.idx] .= 10 .^ logeta

        cmap = cgrad(:jet, rev = true)
        crange = (20, 21.2)
        fig = Figure()
        axs = [Axis(fig[1,i], aspect = DataAspect()) for i in 1:2]
        heatmap!(axs[1], log10.(ground_truth), colormap = cmap, colorrange = crange)
        heatmap!(axs[2], log10.(p_estim), colormap = cmap, colorrange = crange)
        Colorbar(fig[2, :], vertical = false, colormap = cmap, colorrange = crange, width = Relative(0.5))
        save("plots/benchmark5/R=1000km.png", fig)
    end
    mean_log_error = mean( abs.( log10.(ground_truth[paraminv.data.idx]) - logeta ) )
    println(mean_log_error)
    @test mean_log_error < 0.02
end