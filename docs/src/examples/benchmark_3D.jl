#=
After comparing our results against analytical and 1D numerical solution, the obvious next step is to compare our results against a 3D numerical solution. To this end, we reproduce Test 3 from Swierczek-Jereczek et al. (2024), where a 1D Earth structure is perturbed by a Gaussian field in 4 different ways:
1. A reduction of the lithospheric thickness from 150 km (at margin of the domain) to 50 km (at the center of the domain).
2. An increase of the lithospheric thickness from 150 km (at margin of the domain) to 250 km (at the center of the domain).
3. A reduction of the mantle viscosity from 10^21 Pa s (at margin of the domain) to 10^20 Pa s (at the center of the domain).
4. An increase of the mantle viscosity from 10^21 Pa s (at margin of the domain) to 10^22 Pa s (at the center of the domain).

## Case 1: Reduction of lithospheric thickness
=#

using FastIsostasy, CairoMakie, LinearAlgebra

W, n = 3f6, 7
domain = RegionalDomain(W, n)

H_ice_0 = null(domain)
H_ice_1 = 1f3 .* (domain.R .< 1f6)
t_ice = [0, 1f-8, 50f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)

bcs = BoundaryConditions(domain, ice_thickness = it)
sealevel = SeaLevel()

sigma = diagm([(W/4)^2, (W/4)^2])
thinning_lithosphere = generate_gaussian_field(domain, 150f3, [0f0, 0], -100f3, sigma)

solidearth = SolidEarth(
    domain,
    layer_boundaries = thinning_lithosphere,
    layer_viscosities = [1f21],
)
fig = plot_earth(domain, solidearth)

#=
Now let's see what result we obtain:
=#

nout = NativeOutput(vars = [:u, :ue], t = vcat(0, 1f3:1f3:4f3, 5f3, 10f3:10f3:50f3))
sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)
run!(sim)
fig = plot_transect(sim, [:u, :ue])

#=
## Case 2: Increase of lithospheric thickness
=#

thickenning_lithosphere = generate_gaussian_field(domain, 150f3, [0f0, 0], 100f3, sigma)

solidearth = SolidEarth(
    domain,
    layer_boundaries = thickenning_lithosphere,
    layer_viscosities = [1f21],
)
fig = plot_earth(domain, solidearth)

#=
Now let's see what result we obtain:
=#

sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)

run!(sim)
fig = plot_transect(sim, [:u, :ue])

#=
It looks like a thicker lithosphere prevents flexure! This tends to "spread" the deformation by creating a higher lateral coupling between neighbouring cells. In comparison, a thin lithosphere makes the displacement more localized, as the flexural rigidity is lower and the cells are more decoupled from each other.

## Case 3: Reduction of mantle viscosity
=#

log10visc = generate_gaussian_field(domain, 21f0, [0f0, 0], -1f0, sigma)
solidearth = SolidEarth(
    domain,
    layer_boundaries = [150f3],
    layer_viscosities = reshape(10 .^ log10visc, domain.nx, domain.ny, 1),
    calibration = SeakonCalibration(),
)
fig = plot_earth(domain, solidearth)

#=
Looks good
=#

sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)

run!(sim)
fig = plot_transect(sim, [:u, :ue])

#=
## Case 4: Increase of mantle viscosity
=#

log10visc = generate_gaussian_field(domain, 21f0, [0f0, 0], 1f0, sigma)
solidearth = SolidEarth(
    domain,
    layer_boundaries = [150f3],
    layer_viscosities = reshape(10 .^ log10visc, domain.nx, domain.ny, 1),
    calibration = SeakonCalibration(),
)
fig = plot_earth(domain, solidearth)

#=
Looks good
=#

sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)

run!(sim)
fig = plot_transect(sim, [:u, :ue])