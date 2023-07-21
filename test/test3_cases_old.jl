function choose_case(case::String, Omega::ComputationDomain, c::PhysicalConstants)

    # eta = 1.3e21
    # rho = 4.2e3

    maxwelltime_scaling = 1.2
    eta = 1e21
    eta_scaled = maxwelltime_scaling * 1e21
    rho = 4.4e3
    W = (Omega.Wx + Omega.Wy) / 2
    sigma = diagm([(W/4)^2, (W/4)^2])
    prem = load_prem()

    # if case == "gaussian_lo_D"
    #     lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
    #     lb2 = fill(250e3, Omega.Nx, Omega.Ny)
    #     lb = cat(lb1, lb2, dims=3)
    #     p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = [eta, eta],
    #         layer_densities = [rho])
    if case == "gaussian_lo_D"
        lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
        lb2 = fill(151e3, Omega.Nx, Omega.Ny)
        lb3 = fill(400e3, Omega.Nx, Omega.Ny)
        lb4 = fill(700e3, Omega.Nx, Omega.Ny)
        lb5 = fill(1000e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, lb3, lb4, lb5, dims=3)
        lv = fill(1e21, size(lb))
        maxwelltime_scaling!(lv, lb, prem)
        p = MultilayerEarth(Omega, c,
            layer_boundaries = lb, layer_viscosities = lv, layer_densities = [rho])
    elseif case == "gaussian_hi_D"
        lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
        lb2 = fill(250e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = [eta, eta],
            layer_densities = [rho])
    elseif case == "no_litho"
        lb = fill(0.001e3, Omega.Nx, Omega.Ny, 1)
        lv = [eta]
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = lv,
            layer_densities = [rho])
    elseif case == "gaussian_lo_η"
        gauss_visc = maxwelltime_scaling * 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], -1.0, sigma)
        lv = cat(gauss_visc, 3*10 .^ fill(21.0, Omega.Nx, Omega.Ny), dims=3)
        lb1 = fill(150e3, Omega.Nx, Omega.Ny)
        lb2 = fill(1000e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = lv, layer_densities = [rho])
    elseif case == "gaussian_hi_η"
        gauss_visc = maxwelltime_scaling * 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], 1.0, sigma)
        lv = cat(gauss_visc, 3*10 .^ fill(21.0, Omega.Nx, Omega.Ny), dims=3)
        lb1 = fill(150e3, Omega.Nx, Omega.Ny)
        lb2 = fill(1000e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = lv, layer_densities = [rho])
    end

    return p
end