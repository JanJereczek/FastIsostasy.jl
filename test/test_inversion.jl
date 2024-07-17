
function benchmark5()
    Omega = ComputationDomain(3000e3, 5)
    c = PhysicalConstants()
    lb = [88e3, 100e3, 200e3, 300e3]
    dims, logeta, logeta_itp = load_wiens2022(extrapolation_bc = Flat())
    lv = 10 .^ cat([logeta_itp.(Omega.X, Omega.Y, z) for z in lb]..., dims=3)
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    R, H = 1000e3, 1e3
    Hcylinder = uniform_ice_cylinder(Omega, R, H, center = [-1000e3, -1000e3])
    t_Hice = [0.0, 1e-8, t_out[end]]
    Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]
    t_out = years2seconds.(1e3:1e3:2e3)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice)
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