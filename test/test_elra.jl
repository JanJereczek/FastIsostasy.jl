@testset "ELRA" begin
    n = 6

    T = Float32
    W = T(3000e3)               # half-length of the square domain (m)
    domain = RegionalDomain(W, n, use_cuda = use_cuda, correct_distortion = false)
    c = PhysicalConstants()
    p = SolidEarthParameters(domain, tau = years2seconds(3e3), layer_boundaries = [100e3, 600e3],
        rho_litho = 0.0)

    opts = SolverOptions(verbose = true, deformation_model = :elra)
    sim = Simulation(domain, c, p, zeros(2), zeros(2),
        [domain.null, domain.null], opts = opts, output = "sparse")
    now = sim.now

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    now.H_ice = uniform_ice_cylinder(domain, R, H)
    update_loadcolumns!(sim)
    columnanom_load!(sim)
    update_deformation_rhs!(sim, sim.now.u)

    samesize_conv!(
        now.u_eq,
        - (now.columnanoms.load + now.columnanoms.litho) .* sim.c.g .* sim.domain.K .^ 2,
        sim.tools.viscous_convo,
        sim.domain,
    )
    @test 4 < maximum(now.u_eq) < 10
    @test -250 > minimum(now.u_eq) > -290
end