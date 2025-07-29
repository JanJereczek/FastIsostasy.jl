#=
To make things a bit more interesting, we now propose to reproduce one of experiments proposed in a benchmark study of 1D GIA models (Spada et al. 2011). Test 1/2 (results shown in Fig. 9 and 10) consists of a 1D ice cap with a maximum thickness of 1500 m and a maximum latitude of 10° (i.e. the ice cap is roughly 1000 km wide). The lithosphere is assumed to be 100 km thick and the mantle is assumed to be layered, with a viscosity of 1e21 Pa s in the upper mantle and 2e21 Pa s in the lower mantle. The lithosphere is assumed to be elastic, with a Young's modulus of 70 GPa and a Poisson's ratio of 0.28. The ice load is applied at time t = 0 and kept constant until t = 100 kyr. The gravitational response and the resulting change in sea-surface elevation is computed, however without affecting the load that is applied to the solid Earth.

Some of the parameter choices differ from FastIsostasy's default but are set via `PhysicalConstants` and `SolidEarth`.
=#

using FastIsostasy, CairoMakie

W, n, T = 3f6, 7, Float32
domain = RegionalDomain(W, n)
c = PhysicalConstants{T}(rho_ice = 0.931e3)

# Load: 0 at the beginning; ice cap with max latitude 10° and thickness 1500 m afterwards
H_ice_0 = null(domain)
alpha = T(10)
Hmax = T(1500)
H_ice_1 = stereo_ice_cap(domain, alpha, Hmax)
fig = plot_load(domain, H_ice_1)

# Wrap the ice load as a time interpolator and pass it to the boundary conditions.
t_ice = [-1f-3, 0, 100f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)
bcs = BoundaryConditions(domain, ice_thickness = it)

# Activate the computation of the gravitational response.
sealevel = SeaLevel(surface = LaterallyVariableSeaSurface())

# Use the same solid Earth parameters as in Spada et al. (2011).
G = 0.50605f11              # shear modulus (Pa)
nu = 0.28f0
E = G * 2 * (1 + nu)
lb = c.r_equator .- [6301f3, 5951f3, 5701f3]
solidearth = SolidEarth(
    domain,
    layer_boundaries = T.(lb),
    layer_viscosities = [1f21, 1f21, 2f21],
    litho_youngmodulus = E,
    litho_poissonratio = nu,
    rho_litho = 3100f0,
    rho_uppermantle = 3500f0,
)

# Define the fields and time steps to be saved.
nout = NativeOutput(vars = [:u, :ue, :dz_ss],
    t = [0, 10, 1_000, 2_000, 5_000, 10_000, 100_000f0])


sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)
run!(sim)

# Plot the transect of the viscous displacement, the elastic displacement, and the sea-surface elevation change.
fig = plot_transect(sim, [:ue, :u, :dz_ss])

#=
By comparing these results to Fig. 9 of Spada et al. (2011), we can see that the deformational and gravitational response obtained by FastIsostasy are very similar to that obtained by 1D GIA models. Let's see how much time was required for this:
=#

println("Computation time: $(sim.nout.computation_time)")

#=
This is at least 2 orders of magnitude faster than typical 1D GIA models, without any major loss in accuracy!
=#
