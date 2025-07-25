using FastIsostasy

# Set high resolution because implicit time stepping scales well!
W, n = 3f6, 8
domain = RegionalDomain(W, n, correct_distortion = false)

H_ice_0 = kernelnull(domain)
H_ice_1 = 1f3 .* (domain.R .< 1f6)
t_ice = [0, 1, 50f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)

bcs = BoundaryConditions(
    domain,
    ice_thickness = it,
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
)
model = Model(
    lithosphere = RigidLithosphere(),     # Maxwell + LVL: need to define constant time step
    mantle = MaxwellMantle(),
)
params = SolidEarthParameters(
    domain,
    layer_boundaries = [88f3],       #[88f3, 88f3 + 75f3],
    layer_viscosities = [1f21],      #[0.04 * 1f21, 1f21],
)
opts = SolverOptions(diffeq = DiffEqOptions(alg = Euler(), dt_min = 100f0))
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y], t = [1f2, 5f2, 15f2, 5f3, 1f4, 5f4])
tspan = (0f0, 50f3)
sim = Simulation(domain, model, params, tspan; bcs = bcs, nout = nout, opts = opts)
run!(sim)
println("Computation time: $(sim.nout.computation_time)")

using CairoMakie, LinearAlgebra

fig = plot_transect(sim, [:u])