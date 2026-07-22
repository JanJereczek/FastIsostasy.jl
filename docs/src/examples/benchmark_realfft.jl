#=
# Acceleration via Real-FFT backend

The spectral step can be computed with either of two FFT backends. The default,
`ComplexFFTBackend`, uses complex-valued plans (`plan_fft` / `plan_ifft`) over
`(nx, ny)` complex arrays. `RealFFTBackend` uses real-valued plans (`plan_rfft` /
`plan_irfft`) instead: because the displacement and load fields are real, the
non-redundant half of the spectrum is of size `(nx÷2+1, ny)`, which roughly halves
the memory and arithmetic cost of every spectral step.

This is a purely *numerical* choice and is orthogonal to the rheology — both runs
below use exactly the same `SolidEarth` with a `ViscousMantle`, and differ only in
the `fft` field of `SolverOptions`.

This example verifies that both backends give identical results and measures the
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

solidearth = SolidEarth(
    domain,
    mantle = ViscousMantle(),
    lithosphere = LaterallyVariableLithosphere(),
    layer_boundaries = [88f3],
    layer_viscosities = [1f21],
)

#=
Run the simulation with the default complex-FFT backend.
=#

opts_complex = SolverOptions(
    integ = RKCIntegrator(),
    fft = ComplexFFTBackend(),
)
sim_complex = Simulation(domain, bcs, sealevel, solidearth, (0, 50f3);
    nout = nout, opts = opts_complex)
run!(sim_complex)

#=
Run the same simulation with the real-FFT backend. Only `opts` changes.
=#

opts_real = SolverOptions(
    integ = RKCIntegrator(),
    fft = RealFFTBackend(),
)
sim_real = Simulation(domain, bcs, sealevel, solidearth, (0, 50f3);
    nout = nout, opts = opts_real)
run!(sim_real)

#=
## Correctness check

The two displacement fields should be numerically identical up to floating-point
round-off. We print the maximum absolute difference across all stored snapshots.
=#

max_diff = maximum(maximum(abs, u_c .- u_r)
    for (u_c, u_r) in zip(sim_complex.nout.vals[:u], sim_real.nout.vals[:u]))
println("Max |u_complex - u_real| across all snapshots: $max_diff m")

#=
## Timing comparison

The `sim.timer.t_computation` vector stores the wall time (in seconds) elapsed up to
each output snapshot.
=#

t_comp_complex = sim_complex.timer.t_computation
t_comp_real    = sim_real.timer.t_computation

println("Total computation time — ComplexFFTBackend: $(round(t_comp_complex[end]; digits=3)) s")
println("Total computation time — RealFFTBackend:    $(round(t_comp_real[end];    digits=3)) s")
println("Speed-up: $(round(t_comp_complex[end] / t_comp_real[end]; digits=2))×")