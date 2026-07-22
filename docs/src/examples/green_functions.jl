#=
## Green functions

In all examples shown so far, the elastic displacement and the gravitational response were computed by convolving the load with suitable Green's functions. [adhikari](@citet) suggested to expand this idea to the computation of the viscous displacement. This excludes the possibility of including lateral variability without significant improvement of the performance compared to the direct solution of the PDEs. Therefore, we offer this implementation for comparison purposes, but it is not recommended for real applications.
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
sealevel = RegionalSeaLevel()

# solidearth = SolidEarth(
#     domain,
#     tau = 3f3,
#     lithosphere = AdhikariLithosphere(),
#     mantle = AdhikariMantle(),
# )
# nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice],
#     t = [0, 1f2, 3f2, 1f3, 3f3, 1f4, 2f4, 5f4])

# sim = Simulation(domain, bcs, sealevel, solidearth, (0, 50f3); nout = nout)
# run!(sim)
# println("Took $(sim.timer.t_computation[end]) seconds!")

# fig = plot_transect(sim, [:u])