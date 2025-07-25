using FastIsostasy, LinearAlgebra, CairoMakie

T, W, n = Float32, 3f6, 7
domain = RegionalDomain(W, n, correct_distortion = false)

H_ice_0 = kernelnull(domain)
H_ice_1 = 1f3 .* (domain.R .< 1f6)
t_ice = [0, 1, 100f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)
bcs = BoundaryConditions(domain, ice_thickness = it)

model = Model(
    lithosphere = RigidLithosphere(),
    mantle = RelaxedMantle(),
)

sep = SolidEarthParameters(domain, tau = 3f3)
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice],
    t = [0, 1f2, 3f2, 1f3, 3f3, 1f4, 3f4])
tspan = extrema(nout.t)

sim = Simulation(domain, model, sep, tspan; bcs = bcs, nout = deepcopy(nout))
run!(sim)
println("Took $(sim.nout.computation_time) seconds!")

fig = plot_transect(sim, [:u])

#######

# Make this more complex with LVELRA
sigma = diagm([(W/4)^2, (W/4)^2])
log10visc = generate_gaussian_field(domain, 21f0, [0f0, 0], -1f0, sigma)
heatmap(log10visc)
τ1 = get_relaxation_time_weaker.(10 .^ log10visc)
sep1 = SolidEarthParameters(domain, tau = T.(τ1))
sim1 = Simulation(domain, model, sep1, tspan; bcs = bcs, nout = deepcopy(nout))
run!(sim1)
fig1 = plot_transect(sim1, [:u])

τ2 = get_relaxation_time_stronger.(10 .^ log10visc)
sep2 = SolidEarthParameters(domain, tau = T.(τ2))
sim2 = Simulation(domain, model, sep2, tspan; bcs = bcs, nout = deepcopy(nout))
run!(sim2)
fig2 = plot_transect(sim2, [:u])