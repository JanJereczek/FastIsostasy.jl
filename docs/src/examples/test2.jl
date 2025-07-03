using FastIsostasy, TerminalLoggers

W, n, T = 3f6, 8, Float32
use_cuda = false
Omega = RegionalComputationDomain(W, n, correct_distortion = true)
c = PhysicalConstants(rho_ice = 0.931e3)
z_b = fill(1f6, Omega.nx, Omega.ny)   # elevated bedrock to prevent any load from ocean

case = "disc"
H_ice_0 = kernelnull(Omega)
if occursin("disc", case)
    alpha = T(10)                       # max latitude (°) of uniform ice disc
    Hmax = T(1000)                      # uniform ice thickness (m)
    R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
    H_ice_1 = stereo_ice_cylinder(Omega, R, Hmax)
elseif occursin("cap", case)
    alpha = T(10)                       # max latitude (°) of ice cap
    Hmax = T(1500)
    H_ice_1 = stereo_ice_cap(Omega, alpha, Hmax)
end
t_ice = [0, 1, 100f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, Omega)


bcs = ProblemBCs(
    Omega,
    ice_thickness = it,
    sea_surface_elevation = LaterallyVariableSeaSurfaceElevation(),
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
    sea_surface_perturbation = BorderBC(ExtendedBCSpace(), 0f0),
)
sem = SolidEarthModel(
    LaterallyVariableLithosphere(),     # Maxwell + LVL: need to define constant time step
    MaxwellMantle(),
)

G = 0.50605f11              # shear modulus (Pa)
nu = 0.28f0
E = G * 2 * (1 + nu)
lb = c.r_equator .- [6301f3, 5951f3, 5701f3]
sep = SolidEarthParameters(
    Omega,
    layer_boundaries = T.(lb),
    layer_viscosities = [1f21, 1f21, 2f21],
    litho_youngmodulus = E,
    litho_poissonratio = nu,
)

nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y, :dudt],
    t = [0, 10, 1_000, 2_000, 5_000, 10_000, 100_000f0])
opts = SolverOptions(diffeq = DiffEqOptions(alg = BS3(), reltol = 1f-5))
fip = FastIsoProblem(Omega, sem, sep; bcs = bcs, nout = nout, opts = opts)
solve!(fip)
println("Computation time: $(fip.nout.computation_time)")

using CairoMakie, LinearAlgebra

fig = plot_transect(fip, [:ue, :u, :dudt, :dz_ss])