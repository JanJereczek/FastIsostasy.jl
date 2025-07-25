using FastIsostasy

# Set high resolution because implicit time stepping scales well!
W, n = 3f6, 9
Omega = RegionalDomain(W, n, correct_distortion = false)

H_ice_0 = kernelnull(Omega)
H_ice_1 = 1f3 .* (Omega.R .< 1f6)
t_ice = [0, 1, 50f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, Omega)

bcs = BoundaryConditions(
    Omega,
    ice_thickness = it,
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
)
model = Model(
    lithosphere = RigidLithosphere(),     # Maxwell + LVL: need to define constant time step
    mantle = MaxwellMantle(),
)
params = SolidEarthParameters(Omega, rho_litho = 0f0)
params.effective_viscosity .= 1f21
opts = SolverOptions(diffeq = DiffEqOptions(alg = Euler(), dt_min = 100f0))
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y], t = [1f2, 5f2, 15f2, 5f3, 1f4, 5f4])
tspan = (0f0, 50f3)
sim = Simulation(Omega, model, params, tspan; bcs = bcs, nout = nout, opts = opts)
run!(sim)
println("Computation time: $(sim.nout.computation_time)")

using CairoMakie, LinearAlgebra

fig = plot_transect(sim, [:u])