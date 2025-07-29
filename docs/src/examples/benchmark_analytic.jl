#=
In this example, we want to compute the viscous displacement of the upper mantle resulting from a cylindrical ice load with a radius of 1000 km and a thickness of 1 km. To do so, we first generate the computation domain and the load:
=#

using FastIsostasy, CairoMakie

# Create a square domain with size 2*W and 2^n points in each dimension
W, n = 3f6, 7
domain = RegionalDomain(W, n)

# Load: 0 at the beginning; cylinder of radius 1000 km and thickness 1 km afterwards
H_ice_0 = null(domain)
H_ice_1 = 1f3 .* (domain.R .< 1f6)
fig = plot_load(domain, H_ice_1)

#=
This looks good! `plot_load` is a function of FastIsostasy that, similar to all functions `plot_*`, is only loaded if the user is `using Makie`. These functions are here to help the user visualize quickly the results that were obtained and are summarized in []().

First, we wrap the ice load as a time interpolator. Then we define modelling choices (here kept as default), a solid earth structure, an output object, and a time span for the simulation, subsequently executed with `run!()`. The Earth structure deserves particular attention, since it determines key properties of the problem. We here define a single layer boundary at 88 km depth. This means that the lithosphere is 88 km thick and the mantle below it is assumed to be Maxwellian with a viscosity of 1f21 Pa s. The lithosphere density is set to 0, such that it does not contribute to the isostasy.
=#

# Wrap the ice load as a time interpolator
t_ice = [0, 1, 50f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)

# Pass the ice load to the boundary conditions and use default model choices
bcs = BoundaryConditions(domain, ice_thickness = it)

# Use the same geometry as in Bueler et al. (2007).
solidearth = SolidEarth(
    domain,
    layer_boundaries = [88f3],
    layer_viscosities = [1f21],
    rho_litho = 0f0,
)
sealevel = SeaLevel()

# Only store the viscous displacement field at the specified time steps.
nout = NativeOutput(vars = [:u], t = [100, 500, 1500, 5000, 10_000, 50_000f0])

# Set the time span for the simulation, then initialize and run it
sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)
run!(sim)

fig = plot_transect(sim, [:u], analytic_cylinder_solution = true)

#=
In this case, the analytic solution is known and corresponds to the dashed lines. It appears that our numerical solution yields very small error compared to this!

FastIsostasy was designed to be computationally efficient. We are therefore particularly interested in the time needed for the computation, which is stored in the `nout` object of the simulation.
=#

println("Computation time using explicit time stepping: $(sim.nout.computation_time)")

#=
This is the (compilation + computation) time that was required to approximate the viscous displacement field over 50 kyr for a domain of 128x128 points!

If the Earth structure is laterally constant (i.e. the lithospheric thickness and the mantle viscosity do not vary in x and y), the performance can be improved by using an implicit time stepping, as derived by Bueler et al. (2007). This can be achieved by specifying the lithosphere as `RigidLithosphere()` or as `LaterallyConstantLithosphere()` and requires to set a fixed time step via the `DiffEqOptions` in the `SolverOptions`:
=#

solidearth = SolidEarth(
    domain,
    lithosphere = RigidLithosphere(),
    layer_boundaries = [88f3],
    layer_viscosities = [1f21],
)
opts = SolverOptions(diffeq = DiffEqOptions(alg = Euler(), dt_min = 100f0))
sim_implicit = Simulation(domain, bcs, sealevel, solidearth; nout = nout, opts = opts)
run!(sim_implicit)
fig = plot_transect(sim_implicit, [:u], analytic_cylinder_solution = true)

#=
The results appear to be comprable with the previously obtained ones! However, the computation time is lower:
=#

println("Computation time using implicit time stepping: $(sim_implicit.nout.computation_time)")

#=
!!! warning
If you are not sure whether your Earth structure is laterally constant, you should not use the implicit time stepping. The results will be wrong if the lithosphere thickness or the mantle viscosity vary in x and y.

All of the computations shown above are performed with `Float32` as floating point precision. It is easy to switch to `Float64`, by passing arguments as such (e.g. `W = 3e6`). This will increase the accuracy of the results, but also the computation time. The default is `Float32`, which is sufficient for most applications.

Of course, this example remains simple. If you want to incrementally increase the complexity of your simulations, go to the next examples!
=#