#=
# Glacial cycle

We now want to provide an example that presents:
- a heterogeneous lithosphere thickness,
- a heterogeneous upper-mantle viscosity,
- a stack of few viscous channels,
- a more elaborate load that evolves over time,
- transient changes of the relative sea-level.

For this we run a glacial cycle of Antarctica with lithospheric thickness and upper-mantle viscosity from [wiens-seismic-2022](@citet) and the ice thickness history from [peltier-comment-2018](@citet). We start by generating a [`RegionalDomain`](@ref) with intermediate resolution for the sake of the example and load the heterogeneous lithospheric from [pan-influence-2022](@citet) thanks to the convenience of [`load_dataset`](@ref):

In a similar way, we can load the log-viscosity field from [pan-influence-2022](@citet) and plot it at about 300 km depth
=#

using CairoMakie, FastIsostasy

N = 140     # corresponds to 50 km resolution
domain = RegionalDomain(3500e3, 3500e3, N, N)
(; Lon, Lat) = domain

#=
We now load the ice thickness history from ICE6G, again helped by the convenience of [`load_dataset`](@ref). We then create a vector of anomaly snapshots, between which FastIsostasy automatically interpolates linearly. To get an idea of ICE6G, the ice thickness anomaly is then visualised at LGM:
=#
(lon, lat, t), Hice, Hitp = load_dataset("ICE6G_D")
Hice_vec = [Hitp.(Lon, Lat, tk) for tk in t]
k_lgm = argmax([mean(Hice_vec[k]) for k in eachindex(Hice_vec)])
plot_load(domain, Hice_vec[k_lgm])

#=
=#

it = TimeInterpolatedIceThickness(t .* 1e3, Hice_vec, domain)
bcs = BoundaryConditions(domain, ice_thickness = it)
sealevel = SeaLevel(
    surface = LaterallyVariableSeaSurface(),
    load = InteractiveSealevelLoad(),
    bsl = PiecewiseConstantBSL(),
)

#=
The number of layers and the depth of viscous half-space are arbitrary parameters that have to be defined by the user. We here use a relatively shallow model (half-space begins at 300 km depth) with 1 equalisation layer and 3 intermediate layers:
To prevent extreme values of the viscosity, we require it to be larger than a minimal value, fixed to be $$10^{16} \, \mathrm{Pa \, s} $$. We subsequently generate a [`SolidEarth`](@ref) that embeds all the information that has been loaded so far:
=#

(_, _), Tpan, Titp = load_dataset("Lithothickness_Pan2022")
Tlitho = Titp.(Lon, Lat) .* 1e3                     # convert from m to km
mindepth = maximum(Tlitho) + 1e3
lb_vec = range(mindepth, stop = 300e3, length = 3)
lb = cat(Tlitho, [fill(lbval, domain.nx, domain.ny) for lbval in lb_vec]..., dims=3)

(_, _, _), _, logeta_itp = load_dataset("Viscosity_Pan2022")
rlb = c.r_equator .- lb
nlb = size(rlb, 3)
lv_3D = 10 .^ cat([logeta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)

eta_lowerbound = 1e16
lv_3D[lv_3D .< eta_lowerbound] .= eta_lowerbound
maskactive = gaussian_smooth(Hice_vec[k_lgm], domain, 0.05, 0) .> 10
solidearth = SolidEarth(
    domain,
    layer_boundaries = lb,
    layer_viscosities = lv_3D,
    maskactive = maskactive,
)
fig = plot_earth(domain, solidearth)

#=
Finally, we define and solve the resulting [`Simulation`](@ref). We hereby choose the `verbose=true` option to track the progress of the computation.
=#

nout = NativeOutput(vars = [:u, :ue, :dz_ss, :z_ss, :H_ice], t = Float32.(it.t_vec))

sim = Simulation(domain, bcs, sealevel, solidearth; nout = nout)
run!(sim)
println("Computation time: ", sim.nout.computation_time)

#=
Ok, that was fast! We visualise three snapshots of displacements that roughly correspond to LGM, the end of the meltwater pulse 1A and the present-day:
=#

copts = (colormap = :PuOr, colorrange = (-400, 400))
fig = plot_out_over_time(sim, :u_tot, [-26f3, -12f3, 0], copts)

#=
The displayed fields are displacement anomalies w.r.t. to the last interglacial, defined as the reference for the ice thickness anomalies. In [swierczek2024fastisostasy](@citet) these computations are performed on a finer grid and show great agreement with a 3D GIA model that runs between 10,000-100,000 slower (however at with advantage of obtaining a global and richer output).
=#