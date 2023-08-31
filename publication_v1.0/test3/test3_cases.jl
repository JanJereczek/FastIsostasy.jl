function choose_case(case::String, Omega::ComputationDomain, c::PhysicalConstants)

    eta = 1e21
    W = (Omega.Wx + Omega.Wy) / 2
    sigma = diagm([(W/4)^2, (W/4)^2])

    if case == "gaussian_lo_D" || case == "gaussian_hi_D" || case == "no_litho"
        if case == "gaussian_lo_D"
            lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
            lb2 = fill(151e3, Omega.Nx, Omega.Ny)
        elseif case == "gaussian_hi_D"
            lb1 = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
            lb2 = fill(251e3, Omega.Nx, Omega.Ny)
        elseif case == "no_litho"
            lb1 = fill(0.001e3, Omega.Nx, Omega.Ny)
            lb2 = fill(151e3, Omega.Nx, Omega.Ny)
        end
        lb = cat(lb1, lb2, dims=3)
        lv = fill(eta, size(lb))
        p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    elseif case == "gaussian_lo_η" || case == "gaussian_hi_η"
        lb1 = fill(150e3, Omega.Nx, Omega.Ny)
        lb = cat(lb1, dims=3)
        if case == "gaussian_lo_η"
            gauss_visc = 10.0 .^ generate_gaussian_field(
                Omega, 21.0, [0.0, 0.0], -1.0, sigma)
        elseif case == "gaussian_hi_η"
            gauss_visc = 10.0 .^ generate_gaussian_field(
                Omega, 21.0, [0.0, 0.0], 1.0, sigma)
        end
        lv = cat([gauss_visc for k in axes(lb, 3)]..., dims=3)
        p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    elseif case == "ref"
        lb1 = fill(100e3, Omega.Nx, Omega.Ny)
        lb2 = fill(670e3, Omega.Nx, Omega.Ny)
        lv1 = fill(0.5e21, Omega.Nx, Omega.Ny)
        lv2 = fill(5e21, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, dims=3)
        lv = cat(lv1, lv2, dims=3)
        p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    end

    return p
end