using FastIsostasy

W, n, T = 3f6, 7, Float32
domain = RegionalDomain(W, n, correct_distortion = true)
c = PhysicalConstants(rho_ice = 0.931e3)
z_b = fill(1f6, domain.nx, domain.ny)   # elevated bedrock to prevent any load from ocean

H_ice_0 = kernelnull(domain)
alpha = T(10)                       # max latitude (°) of ice cap
Hmax = T(1500)
H_ice_1 = stereo_ice_cap(domain, alpha, Hmax)
t_ice = [0, 1, 100f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)

bcs = BoundaryConditions(
    domain,
    ice_thickness = it,
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
    sea_surface_perturbation = BorderBC(ExtendedBCSpace(), 0f0),
)
model = Model(
    lithosphere = LaterallyVariableLithosphere(),     # Maxwell + LVL: need to define constant time step
    mantle = MaxwellMantle(),
    sea_surface = LaterallyVariableSeaSurface(),
)

G = 0.50605f11              # shear modulus (Pa)
nu = 0.28f0
E = G * 2 * (1 + nu)
lb = c.r_equator .- [6301f3, 5951f3, 5701f3]
sep = SolidEarthParameters(
    domain,
    layer_boundaries = T.(lb),
    layer_viscosities = [1f21, 1f21, 2f21],
    litho_youngmodulus = E,
    litho_poissonratio = nu,
)

nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y, :dudt],
    t = [0, 10, 1_000, 2_000, 5_000, 10_000, 100_000f0])
tspan = (0f0, 100_000f0)
sim = Simulation(domain, model, sep, tspan; bcs = bcs, nout = deepcopy(nout))
run!(sim)
println("Computation time: $(sim.nout.computation_time)")

using CairoMakie, LinearAlgebra

fig = plot_transect(sim, [:ue, :u, :dudt, :dz_ss])