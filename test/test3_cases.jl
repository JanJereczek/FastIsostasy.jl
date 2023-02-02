function choose_case(case::String, Omega::ComputationDomain, c::PhysicalConstants)

    if case == "binaryD"
        layer1_begin = generate_window_field(Omega, 100e3, 50e3, 250e3)
        layer2_begin = fill(250e3, Omega.N, Omega.N)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = init_multilayer_earth(Omega, c, layers_begin = lb, layers_viscosity = [1e21, 1e21])
    elseif case == "binaryη"
        window_viscosity = 10 .^ generate_window_field(Omega, 21., 18., 23.)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(window_viscosity, halfspace_viscosity, dims=3)
        p = init_multilayer_earth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    elseif case == "binaryDη"
        layer1_begin = generate_window_field(Omega, 100e3, 50e3, 250e3)
        layer2_begin = fill(250e3, Omega.N, Omega.N)
        layer3_begin = fill(400e3, Omega.N, Omega.N)
        lb = cat(layer1_begin, layer2_begin, layer3_begin, dims=3)

        eqlayer_visc = fill(1e21, Omega.N, Omega.N)
        window_viscosity = 10 .^ generate_window_field(Omega, 21., 19., 23.)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(eqlayer_visc, window_viscosity, halfspace_viscosity, dims=3)

        ld = [3.3e3, 3.3e3]

        p = init_multilayer_earth(
            Omega,
            c,
            layers_begin = lb,
            layers_viscosity = lv,
            layers_density = ld,
        )
    elseif case == "gaussian_lo_D"
        L = (Omega.Lx + Omega.Ly) / 2
        sigma = diagm([(L/4)^2, (L/4)^2])
        layer1_begin = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
        layer2_begin = fill(250e3, Omega.N, Omega.N)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = init_multilayer_earth(Omega, c, layers_begin = lb, layers_viscosity = [1e21, 1e21])
    elseif case == "gaussian_hi_D"
        L = (Omega.Lx + Omega.Ly) / 2
        sigma = diagm([(L/4)^2, (L/4)^2])
        layer1_begin = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
        layer2_begin = fill(250e3, Omega.N, Omega.N)
        lb = cat(layer1_begin, layer2_begin, dims=3)
        p = init_multilayer_earth(Omega, c, layers_begin = lb, layers_viscosity = [1e21, 1e21])
    elseif case == "gaussian_lo_η"
        L = (Omega.Lx + Omega.Ly) / 2
        sigma = diagm([(L/4)^2, (L/4)^2])
        gauss_visc = 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], -2.0, sigma)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(gauss_visc, halfspace_viscosity, dims=3)
        p = init_multilayer_earth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    elseif case == "gaussian_hi_η"
        L = (Omega.Lx + Omega.Ly) / 2
        sigma = diagm([(L/4)^2, (L/4)^2])
        gauss_visc = 10.0 .^ generate_gaussian_field(Omega, 21.0, [0.0, 0.0], 2.0, sigma)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(gauss_visc, halfspace_viscosity, dims=3)
        p = init_multilayer_earth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    end

    return p
end