function choose_case(case::String, Omega::ComputationDomain, c::PhysicalConstants)

    eta = 1e21
    rho = 4.4e3
    W = (Omega.Wx + Omega.Wy) / 2
    sigma = diagm([(W/4)^2, (W/4)^2])
    prem = load_prem()

    if case == "gaussian_lo_D" || case == "gaussian_hi_D" || case == "gaussian_no_D"
        if case == "gaussian_lo_D"
            lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
            lb2 = fill(151e3, Omega.Nx, Omega.Ny)
        elseif case == "gaussian_hi_D"
            lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
            lb2 = fill(251e3, Omega.Nx, Omega.Ny)
        elseif case == "gaussian_no_D"
            lb1 = fill(0.001e3, Omega.Nx, Omega.Ny, 1)
            lb2 = fill(151e3, Omega.Nx, Omega.Ny)
        end
        lb3 = fill(400e3, Omega.Nx, Omega.Ny)
        lb4 = fill(700e3, Omega.Nx, Omega.Ny)
        lb5 = fill(1000e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, lb3, lb4, lb5, dims=3)
        lv = fill(eta, size(lb))
        # maxwelltime_scaling!(lv, lb, prem)
        p = MultilayerEarth(Omega, c,
            layer_boundaries = lb, layer_viscosities = lv, layer_densities = [rho])
    elseif case == "gaussian_lo_η" || case == "gaussian_hi_η"
        lb1 = fill(150e3, Omega.Nx, Omega.Ny)
        lb2 = fill(400e3, Omega.Nx, Omega.Ny)
        lb3 = fill(700e3, Omega.Nx, Omega.Ny)
        lb4 = fill(1000e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, lb3, lb4, dims=3)
        if case == "gaussian_lo_η"
            gauss_visc = 10.0 .^ generate_gaussian_field(
                Omega, 21.0, [0.0, 0.0], -1.0, sigma)
        elseif case == "gaussian_hi_η"
            gauss_visc = 10.0 .^ generate_gaussian_field(
                Omega, 21.0, [0.0, 0.0], 1.0, sigma)
        end
        lv = cat([gauss_visc for k in axes(lb, 3)]..., dims=3)
        # maxwelltime_scaling!(lv, lb, prem)
        p = MultilayerEarth(Omega, c,
            layer_boundaries = lb, layer_viscosities = lv, layer_densities = [rho])
    end

    return p
end