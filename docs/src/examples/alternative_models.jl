using FastIsostasy, LinearAlgebra, CairoMakie

T, W, n = Float32, 3f6, 7
Omega = RegionalComputationDomain(W, n, correct_distortion = true)

H_ice_0 = zeros(T, Omega.nx, Omega.ny)
H_ice_1 = 1f3 .* (Omega.R .< 1f6)
t_ice = [0, 1f-8, 100f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, Omega)

bcs = ProblemBCs(
    Omega,
    ice_thickness = it,
    sea_surface = LaterallyConstantSeaSurface(),
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
    sea_surface_perturbation = BorderBC(ExtendedBCSpace(), 0f0),
)
sem = SolidEarthModel(
    LaterallyConstantLithosphere(),     # Maxwell + LVL: need to define constant time step
    RelaxedMantle(),
)

sep = SolidEarthParameters(Omega, tau = 3f3)
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice],
    t = vcat(0, 1f3:1f3:4f3, 5f3:5f3:50f3))

fip = FastIsoProblem(Omega, sem, sep; bcs = bcs, nout = nout)
solve!(fip)
println("Took $(fip.nout.computation_time) seconds!")

fig = plot_transect(fip, [:u])

#######

# Make this more complex with LVELRA
sigma = diagm([(W/4)^2, (W/4)^2])
log10visc = generate_gaussian_field(Omega, 21f0, [0f0, 0], -1f0, sigma)
heatmap(log10visc)
τ1 = get_relaxation_time_weaker.(10 .^ log10visc)
sep1 = SolidEarthParameters(Omega, tau = T.(τ1))
fip1 = FastIsoProblem(Omega, sem, sep1; bcs = bcs, nout = deepcopy(nout))
solve!(fip1)
fig1 = plot_transect(fip1, [:u])

τ2 = get_relaxation_time_stronger.(10 .^ log10visc)
sep2 = SolidEarthParameters(Omega, tau = T.(τ2))
fip2 = FastIsoProblem(Omega, sem, sep2; bcs = bcs, nout = deepcopy(nout))
solve!(fip2)
fig2 = plot_transect(fip2, [:u])