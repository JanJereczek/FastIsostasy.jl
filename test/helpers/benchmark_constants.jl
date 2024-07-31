function benchmark1_constants(Omega)
    c = PhysicalConstants()
    p = LayeredEarth(Omega, rho_litho = 0.0)
    t_out = [0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0]
    
    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]

    R, H = 1000e3, 1e3
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]

    return c, p, t_out, R, H, t_Hice, Hice
end