using FastIsostasy

function minimal_fip()
    W = 3000e3
    n = 7
    Omega = ComputationDomain(W, n, correct_distortion = false)
    c = PhysicalConstants(rho_litho = 0.0)
    p = LayeredEarth(Omega)

    R = 1000e3
    H = 1e3
    Hice = uniform_ice_cylinder(Omega, R, H)
    Hice_vec = [fill(0.0, Omega.nx, Omega.ny), 0.1 .* Hice]
    tice = [0, 100]
    t_out = [0.0, 200.0, 600.0, 2000.0, 5000.0, 10_000.0, 50_000.0]
    return FastIsoProblem(Omega, c, p, t_out, tice, Hice_vec)
end