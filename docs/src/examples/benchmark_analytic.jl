using FastIsostasy

W, n = 3f6, 7
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
    lithosphere = LaterallyVariableLithosphere(),
    mantle = MaxwellMantle(),
)
params = SolidEarthParameters(
    domain,
    layer_boundaries = [88f3],
    layer_viscosities = [1f21],
    rho_litho = 0f0,
)
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y],
    t = [100, 500, 1500, 5000, 10_000, 50_000f0])
tspan = (0f0, 50_000f0)
sim = Simulation(domain, model, params, tspan; bcs = bcs, nout = nout)
run!(sim)
println("Computation time: $(sim.nout.computation_time)")

using CairoMakie, LinearAlgebra

fig = plot_transect(sim, [:u])