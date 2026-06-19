#=
# Real-FFT Maxwell mantle

`MaxwellMantle` performs the spectral step with complex-valued FFT plans, operating on
`(nx, ny)` complex arrays. `RealMaxwellMantle` uses real-valued FFT plans (`plan_rfft` /
`plan_irfft`) instead. Because the displacement and load fields are real, the non-redundant
half of the spectrum is of size `(nx÷2+1, ny)`, which roughly halves the memory and
arithmetic cost of every spectral step.

This example verifies that both mantles give identical results and measures the
speed-up on a standard benchmark geometry (cylindrical load, same setup as the
[Analytical benchmark](@ref)).
=#

using FastIsostasy, CairoMakie

W, n = 3f6, 9
domain = RegionalDomain(W, n)

H_ice_0 = zeros(domain)
H_ice_1 = 1f3 .* (domain.R .< 1f6)

t_ice  = [0, 1, 50f3]
H_ice  = [H_ice_0, H_ice_1, H_ice_1]
it     = TimeInterpolatedIceThickness(t_ice, H_ice, domain)
bcs    = BoundaryConditions(domain, ice_thickness = it)
sealevel = RegionalSeaLevel()

nout = NativeOutput(vars = [:u],
    t = [100, 500, 1500, 5000, 10_000, 50_000f0])

opts = SolverOptions()

#=
Run the simulation with the standard complex-FFT Maxwell mantle.
=#

solidearth_complex = SolidEarth(
    domain,
    mantle = MaxwellMantle(),
    lithosphere = LaterallyVariableLithosphere(),
    layer_boundaries = [88f3],
    layer_viscosities = [1f21],
)

sim_complex = Simulation(domain, bcs, sealevel, solidearth_complex, (0, 50f3);
    nout = nout, opts = opts)
run!(sim_complex)

#=
Run the same simulation with the real-FFT Maxwell mantle.
=#

solidearth_real = SolidEarth(
    domain,
    mantle = RealMaxwellMantle(),
    lithosphere = LaterallyVariableLithosphere(),
    layer_boundaries = [88f3],
    layer_viscosities = [1f21],
)

sim_real = Simulation(domain, bcs, sealevel, solidearth_real, (0, 50f3);
    nout = nout, opts = opts)
run!(sim_real)

#=
## Correctness check

The two displacement fields should be numerically identical up to floating-point
round-off. We print the maximum absolute difference across all stored snapshots.
=#

max_diff = maximum(maximum(abs, u_c .- u_r)
    for (u_c, u_r) in zip(sim_complex.nout.vals[:u], sim_real.nout.vals[:u]))
println("Max |u_complex - u_real| across all snapshots: $max_diff m")

fig_check = Figure()
ax = Axis(fig_check[1, 1],
    xlabel = "x (m)", ylabel = "Viscous displacement (m)",
    title = "Final snapshot — transect at j = ny÷2")
j = domain.ny ÷ 2
lines!(ax, domain.x, sim_complex.nout.vals[:u][end][:, j], label = "MaxwellMantle")
lines!(ax, domain.x, sim_real.nout.vals[:u][end][:, j],    label = "RealMaxwellMantle",
    linestyle = :dash)
axislegend(ax)
fig_check

#=
## Timing comparison

The `sim.timer.t_computation` vector stores the wall time (in seconds) elapsed up to
each output snapshot.
=#

t_comp_complex = sim_complex.timer.t_computation
t_comp_real    = sim_real.timer.t_computation

println("Total computation time — MaxwellMantle:     $(round(t_comp_complex[end]; digits=3)) s")
println("Total computation time — RealMaxwellMantle: $(round(t_comp_real[end];    digits=3)) s")
println("Speed-up: $(round(t_comp_complex[end] / t_comp_real[end]; digits=2))×")

fig_timing, ax_t, _ = lines(t_comp_complex, sim_complex.timer.t_vec,
    label = "MaxwellMantle")
lines!(ax_t, t_comp_real, sim_real.timer.t_vec,
    label = "RealMaxwellMantle", linestyle = :dash)
ax_t.xlabel = "Computation time (s)"
ax_t.ylabel = "Simulation years"
axislegend(ax_t, position = :lt)
fig_timing
