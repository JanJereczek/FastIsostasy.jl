using FastIsostasy, CairoMakie

# This should rather be in `/publication_v2.0`
function main(L, mode)
    println("Running mode $mode for L = $L")
    W, n, T = 3f6, 7, Float32
    domain = RegionalDomain(W, n, correct_distortion = false)

    H_ice_0 = kernelnull(domain)
    H_ice_1 = 1f3 .* (domain.R .< L)
    t_ice = [0, 1, 100f3]
    H_ice = [H_ice_0, H_ice_1, H_ice_1]
    it = TimeInterpolatedIceThickness(t_ice, H_ice, domain)

    bcs = BoundaryConditions(domain, ice_thickness = it)
    model = Model(lithosphere = RigidLithosphere(), mantle = MaxwellMantle())

    lb = [88f3, 88f3 + 75f3]

    if occursin("3layers", mode)
        println("Using 3 layers...")
        lv = [0.04f0 * 1f21, 1f21]
    else
        println("Using 2 layers...")
        lv = [1f21, 1f21]
    end

    if occursin("meanlog", mode)
        println("Using mean log viscosity lumping...")
        lpg = MeanLogViscosityLumping()
    elseif occursin("mean", mode)
        println("Using mean viscosity lumping...")
        lpg = MeanViscosityLumping()
    elseif occursin("time", mode)
        println("Using time domain viscosity lumping...")
        lpg = TimeDomainViscosityLumping()
    else
        println("Using frequency domain viscosity lumping...")
        lpg = FreqDomainViscosityLumping()
    end

    sep = SolidEarthParameters(domain, layer_boundaries = lb, layer_viscosities = lv,
        calibration = NoCalibration(), compressibility = IncompressibleMantle(),
        lumping = lpg)
    @show extrema(sep.effective_viscosity)

    nout = NativeOutput(vars = [:u], t = collect(0:100:30_000f0))
    tspan = extrema(nout.t)
    opts = SolverOptions(verbose = false)
    sim = Simulation(domain, model, sep, tspan; bcs = bcs, nout = deepcopy(nout), opts = opts)
    run!(sim)
    println("Computation time: $(sim.nout.computation_time)")
    println("----------------------------------------------")

    u_max = maximum(abs.(sim.nout.vals[:u][end]))
    i_tau = findfirst(x -> x > (1-exp(-1)) * u_max, [maximum(abs.(u)) for u in sim.nout.vals[:u]])
    tau = sim.nout.t[i_tau]
    return tau
end

# if we begin lower, the load can't be resolved
L = 4.7:0.1:6.3
L = Float32.(10 .^ L)
modes = ["2layers", "mean3layers", "meanlog3layers", "time3layers", "freq3layers"]

taus = [Float32[] for _ in modes]
for l in L
    for i in eachindex(modes)
        tau = main(l, modes[i])
        push!(taus[i], tau)
    end
end

set_theme!(theme_latexfonts())
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1])
for i in eachindex(modes)
    scatterlines!(ax, log10.(L), taus[i], label = modes[i])
end
ax.xlabel = "Ice radius (km)"
ax.ylabel = "Relaxation time (yr)"
axislegend(ax, position = :rt)
fig
save("src/assets/benchmark_timescales.png", fig)