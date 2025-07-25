function benchmark1_sim(T, use_cuda)

    W, n = T(3f6), 7
    domain = RegionalDomain(W, n, correct_distortion = false, use_cuda = use_cuda)

    H_ice_0 = kernelnull(domain)
    H_ice_1 = T.(1f3 .* (domain.R .< 1f6))
    t_ice = T.([0, 1, 50f3])
    H_ice = [H_ice_0, H_ice_1, H_ice_1]
    it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)

    bcs = BoundaryConditions(
        domain,
        ice_thickness = it,
        viscous_displacement = BorderBC(RegularBCSpace(), T.(0f0)),
        elastic_displacement = BorderBC(ExtendedBCSpace(), T.(0f0)),
    )
    model = Model(
        lithosphere = RigidLithosphere(),     # Maxwell + LVL: need to define constant time step
        mantle = MaxwellMantle(),
    )
    params = SolidEarthParameters(domain, rho_litho = T.(0f0))
    params.effective_viscosity .= 1f21
    nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y],
        t = [100, 500, 1500, 5000, 10_000, 50_000f0])
    tspan = T.((0, 50_000))
    sim = Simulation(domain, model, params, tspan; bcs = bcs, nout = nout)

    return sim
end