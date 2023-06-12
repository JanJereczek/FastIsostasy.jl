function choose_case(case::String, Omega::ComputationDomain, c::PhysicalConstants)

    eta = 1e21
    rho = 4.4e3
    if case == "gaussian_lo_D"
        W = (Omega.Wx + Omega.Wy) / 2
        sigma = diagm([(W/4)^2, (W/4)^2])
        layer1_begin = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
        layer2_begin = fill(250e3, Omega.N, Omega.N)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = [eta, eta],
            layers_density = [rho])
    elseif case == "gaussian_hi_D"
        W = (Omega.Wx + Omega.Wy) / 2
        sigma = diagm([(W/4)^2, (W/4)^2])
        layer1_begin = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
        layer2_begin = fill(250e3, Omega.N, Omega.N)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = MultilayerEarth(Omega, c, layer_boundaries = lb, layer_viscosities = [eta, eta],
            layers_density = [rho])
    elseif case == "gaussian_lo_η"
        W = (Omega.Wx + Omega.Wy) / 2
        sigma = diagm([(W/4)^2, (W/4)^2])
        gauss_visc = 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], -1.0, sigma)
        lv = cat(gauss_visc, dims=3)
        p = MultilayerEarth(Omega, c, layer_viscosities = lv)
    elseif case == "gaussian_hi_η"
        W = (Omega.Wx + Omega.Wy) / 2
        sigma = diagm([(W/4)^2, (W/4)^2])
        gauss_visc = 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], 1.0, sigma)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(gauss_visc, halfspace_viscosity, dims=3)
        p = MultilayerEarth(Omega, c, layer_viscosities = lv)
    end

    return p
end