using LinearAlgebra

function choose_case(case::String, Omega::ComputationDomain)

    eta = 1e21
    W = (Omega.Wx + Omega.Wy) / 2
    sigma = diagm([(W/4)^2, (W/4)^2])

    if case == "homogeneous"
        lv = fill(eta, Omega.Nx, Omega.Ny)
        litho_thickness = 150e3
        layering = UniformLayering(1, [litho_thickness])
    elseif case == "gaussian_lo_D" || case == "gaussian_hi_D" || case == "no_litho"
        if case == "gaussian_lo_D"
            litho_thickness = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], -100e3, sigma)
            lb2 = 151e3
            layering = EqualizedLayering(2, [88e3, lb2])
        elseif case == "gaussian_hi_D"
            litho_thickness = generate_gaussian_field(Omega, 150e3, [0.0, 0.0], 100e3, sigma)
            lb2 = 251e3
            layering = EqualizedLayering(2, [88e3, lb2])
        elseif case == "no_litho"
            litho_thickness = 0.001e3
            lb2 = 151e3
            layering = EqualizedLayering(2, [litho_thickness, lb2])
        end
        lb = cat(lb1, lb2, dims=3)
        lv = fill(eta, size(lb))
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
    elseif case == "ref"
        lb1 = fill(100e3, Omega.Nx, Omega.Ny)
        lb2 = fill(670e3, Omega.Nx, Omega.Ny)
        lv1 = fill(0.5e21, Omega.Nx, Omega.Ny)
        lv2 = fill(5e21, Omega.Nx, Omega.Ny)
        lb = cat(lb1, lb2, dims=3)
        lv = cat(lv1, lv2, dims=3)
    end
    p = LayeredEarth(Omega, U, layering = EqualizedLayering(2, [88e3, lb2]), layer_viscosities = lv,
        rho_uppermantle = 3.6e3, litho_thickness = lb1)

    return p, lb, lv
end