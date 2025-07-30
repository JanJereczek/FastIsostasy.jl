#=
# Alternative models

## ELRA

Sometimes people might want to use ELRA (LeMeur & Huybrechts, 1996):
=#

using FastIsostasy, CairoMakie

T, W, n = Float32, 3f6, 7
domain = RegionalDomain(W, n, correct_distortion = false)

H_ice_0 = zeros(domain)
H_ice_1 = 1f3 .* (domain.R .< 1f6)
t_ice = [0, 1, 5f4]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)
bcs = BoundaryConditions(domain, ice_thickness = it)
sealevel = SeaLevel()

solidearth = SolidEarth(
    domain,
    tau = 3f3,
    lithosphere = RigidLithosphere(),
    mantle = RelaxedMantle(),
)
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice],
    t = [0, 1f2, 3f2, 1f3, 3f3, 1f4, 2f4, 5f4])

sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)
run!(sim)
println("Took $(sim.nout.computation_time) seconds!")

fig = plot_transect(sim, [:u])

#=

## ELRA with 2D relaxation time

Sometimes people might want to use ELRA with 2D maps of the relaxation time (Van Calcar et al., 2025)
=#
using LinearAlgebra

sigma = diagm([(W/4)^2, (W/4)^2])
log10visc = generate_gaussian_field(domain, 21f0, [0f0, 0], -1f0, sigma)
heatmap(log10visc)
τ_weak = get_relaxation_time_weaker.(10 .^ log10visc)
println("Extrema of weak 2D relaxation time: $(extrema(τ_weak))")

solidearth_weak = SolidEarth(
    domain,
    tau = T.(τ_weak),
    lithosphere = RigidLithosphere(),
    mantle = RelaxedMantle(),
)
sim_weak = Simulation(domain, bcs, sealevel, solidearth_weak; nout = nout)
run!(sim_weak)
fig = plot_transect(sim_weak, [:u])

#=
Alternatively:
=#

τ_strong = get_relaxation_time_stronger.(10 .^ log10visc)
println("Extrema of strong 2D relaxation time: $(extrema(τ_strong))")

solidearth_strong = SolidEarth(
    domain,
    tau = T.(τ_strong),
    lithosphere = RigidLithosphere(),
    mantle = RelaxedMantle(),
)
sim_strong = Simulation(domain, bcs, sealevel, solidearth_strong; nout = nout)
run!(sim_strong)
fig = plot_transect(sim_strong, [:u])