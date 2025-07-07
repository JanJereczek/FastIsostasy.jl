using FastIsostasy

W, n = 3f6, 7
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
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y],
    t = [100, 500, 1500, 5000, 10_000, 50_000f0])
tspan = (0f0, 50_000f0)
fip = Simulation(Omega, model, params, tspan; bcs = bcs, nout = nout)
run!(fip)
println("Computation time: $(fip.nout.computation_time)")

using CairoMakie, LinearAlgebra

fig = plot_transect(fip, [:u])