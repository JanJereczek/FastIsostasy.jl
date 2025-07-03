using FastIsostasy, TerminalLoggers

W, n = 3f6, 8
use_cuda = false
Omega = RegionalComputationDomain(W, n, use_cuda = use_cuda)

H_ice_0 = kernelnull(Omega)
H_ice_1 = 1f3 .* (Omega.R .< 1f6)
t_ice = [0, 1, 50f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, Omega)

bcs = ProblemBCs(
    Omega,
    ice_thickness = it,
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
)
sem = SolidEarthModel(
    RigidLithosphere(),     # Maxwell + LVL: need to define constant time step
    MaxwellMantle(),
)
sep = SolidEarthParameters(Omega, rho_litho = 0f0)
sep.effective_viscosity .= 1f21
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y],
    t = [0, 100, 500, 1500, 5000, 10_000, 50_000f0])
fip = FastIsoProblem(Omega, sem, sep; bcs = bcs, nout = nout)
solve!(fip)
println("Computation time: $(fip.nout.computation_time)")

using CairoMakie

fig = plot_transect(fip, [:u])