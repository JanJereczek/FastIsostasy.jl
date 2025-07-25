using FastIsostasy

W, n, T = 3f6, 8, Float32
domain = RegionalComputationDomain(W, n)
(; Lon, Lat) = domain

(_, _), Tpan, Titp = load_dataset("Lithothickness_Pan2022")
Tlitho = Titp.(Lon, Lat) .* 1e3                     # convert from m to km

function nicer_heatmap(X)
    fig = Figure(size = (800, 700))
    ax = Axis(fig[1, 1], aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, X)
    Colorbar(fig[1, 2], hm, height = Relative(0.6))
    return fig
end
nicer_heatmap(Tlitho)

(_, _, _), _, logeta_itp = load_dataset("Viscosity_Pan2022")
logeta300 = logeta_itp.(Lon, Lat, c.r_equator - 300e3)
nicer_heatmap(logeta300)

layering = ParallelLayering()
lb = get_layer_boundaries(domain, T.(Tlitho), layering)

rlb = c.r_equator .- lb
nlb = size(rlb, 3)
lv_3D = 10 .^ cat([logeta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)

eta_lowerbound = 1e16
lv_3D[lv_3D .< eta_lowerbound] .= eta_lowerbound
p = SolidEarthParameters(domain, layer_boundaries = lb, layer_viscosities = T.(lv_3D))
nicer_heatmap(log10.(p.effective_viscosity))

(lon, lat, t), Hice, Hitp = load_dataset("ICE6G_D")
Hice_vec = [T.(Hitp.(Lon, Lat, tk)) for tk in t]
nicer_heatmap(Hitp.(Lon, Lat, -26) - Hitp.(Lon, Lat, 0))
it = TimeInterpolatedIceThickness(t .* 1f3, Hice_vec, domain)

bcs = ProblemBCs(
    domain,
    ice_thickness = it,
    sea_surface = LaterallyVariableSeaSurface(),
    viscous_displacement = BorderBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
    sea_surface_perturbation = BorderBC(ExtendedBCSpace(), 0f0),
)
sem = SolidEarthModel(
    LaterallyVariableLithosphere(),     # Maxwell + LVL: need to define constant time step
    MaxwellMantle(),
)

nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y, :dudt], t = t .* 1f3)
opts = SolverOptions(diffeq = DiffEqOptions(alg = BS3(), reltol = 1f-5))

sim = Simulation(domain, sem, sep; bcs = bcs, nout = nout, opts = opts)
run!(sim)
println(sim.nout.computation_time)

#=
For a resolution of 50 km, the computation time of this last step takes about 30 seconds on a modern i7 (Intel i7-10750H CPU @ 2.60GHz)! We visualise three snapshots of displacements that roughly correspond to LGM, the end of meltwater pulse 1A and the present-day:
=#

tplot = [-26f3, -12f3, 0]
fig = Figure(size = (1200, 400))
opts = ( colormap = :PuOr, colorrange = (-400, 400) )
for k in eachindex(tplot)
    kfi = argmin( abs.(tplot[k] .- sim.nout.t) )
    ax = Axis(fig[1, k], aspect = DataAspect(), title = "t = $(sim.nout.t[kfi]) kyr")
    hidedecorations!(ax)
    heatmap!(ax, sim.nout.vals[:u][kfi] + sim.nout.vals[:ue][kfi]; opts...)
    println(kfi)
end
Colorbar(fig[1, 4], height = Relative(0.6); opts...)
fig