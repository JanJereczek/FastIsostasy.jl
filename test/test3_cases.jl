function choose_case(case::String, Omega::ComputationDomain, c::PhysicalConstants)

    # eta = 1.3e21
    # rho = 4.2e3

    eta = 1.2e21
    rho = 4.4e3
    W = (Omega.Wx + Omega.Wy) / 2
    sigma = diagm([(W/4)^2, (W/4)^2])

    if case == "gaussian_lo_D"
        layer1_begin = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
        layer2_begin = fill(250e3, Omega.Nx, Omega.Ny)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = [eta, eta],
            layers_density = [rho])
    elseif case == "gaussian_hi_D"
        layer1_begin = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
        layer2_begin = fill(250e3, Omega.Nx, Omega.Ny)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = [eta, eta],
            layers_density = [rho])
    elseif case == "gaussian_no_D"
        lb = fill(0.001e3, Omega.Nx, Omega.Ny, 1)
        lv = [eta]
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = lv,
            layers_density = [rho])
    elseif case == "gaussian_lo_η"
        gauss_visc = 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], -1.0, sigma)
        lv = cat(gauss_visc, dims=3)
        p = MultilayerEarth(Omega, c, layer_viscosities = lv)
    elseif case == "gaussian_hi_η"
        gauss_visc = 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], 1.0, sigma)
        halfspace_viscosity = fill(1e21, Omega.Nx, Omega.Ny)
        lv = cat(gauss_visc, halfspace_viscosity, dims=3)
        p = MultilayerEarth(Omega, c, layer_viscosities = lv)
    end

    return p
end