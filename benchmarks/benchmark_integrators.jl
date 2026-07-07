# =============================================================================
# Benchmark of the built-in FIAlgorithm time-stepping backend on the analytic
# cylinder problem (same setup as docs/src/examples/benchmark_analytic.jl).
#
# Records, per algorithm: wall-clock time of `run!` (best of a few reps,
# post-warmup), allocations, number of RHS evaluations, and the final peak
# viscous displacement.
#
# The RHS evaluation count is read off `sim.timer.t_computation`, which is
# pushed exactly once per call to `update_diagnostics!` (the RHS). Since the RHS
# (FFTs + convolutions) dominates the per-step cost, this is the fairest metric.
#
# The head-to-head against the former OrdinaryDiffEq backend that motivated this
# implementation is recorded in docs/src/integrators.md.
#
# Run with:  julia --project=. benchmarks/benchmark_integrators.jl
# =============================================================================

using FastIsostasy
using Printf

const W, n = 3f6, 7                      # 2W-wide square domain, 2^n points/side
const TSPAN = (0f0, 50f3)
const SAVE_T = [100, 500, 1500, 5000, 10_000, 50_000f0]

function build_sim(alg; reltol = 1f-5, dt_min = nothing)
    domain = RegionalDomain(W, n)

    H_ice_0 = zeros(domain)
    H_ice_1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0, 1, 50f3], [H_ice_0, H_ice_1, H_ice_1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)

    solidearth = SolidEarth(domain,
        layer_boundaries = [88f3], layer_viscosities = [1f21], rho_litho = 0f0)
    sealevel = RegionalSeaLevel()
    nout = NativeOutput(vars = [:u], t = SAVE_T)

    opts = SolverOptions(verbose = false,
        diffeq = DiffEqOptions(alg = alg, reltol = reltol, dt_min = dt_min))
    return Simulation(domain, bcs, sealevel, solidearth, TSPAN; nout = nout, opts = opts)
end

nrhs(sim) = length(sim.timer.t_computation)   # ~ number of RHS evaluations

function measure(alg; reltol = 1f-5, dt_min = nothing, nrep = 3)
    run!(build_sim(alg; reltol = reltol, dt_min = dt_min))   # warm-up (compile)

    best_time = Inf
    allocs = 0
    rhs = 0
    peak = 0.0f0
    for _ in 1:nrep
        sim = build_sim(alg; reltol = reltol, dt_min = dt_min)
        stats = @timed run!(sim)
        if stats.time < best_time
            best_time = stats.time
            allocs = stats.bytes
            rhs = nrhs(sim)
            peak = maximum(abs, sim.now.u)
        end
    end
    return (; time = best_time, bytes = allocs, rhs = rhs, peak = peak)
end

const RELTOL = 1f-5
algs = [
    ("FIBS3",   FIBS3(),   nothing),
    ("FITsit5", FITsit5(), nothing),
    ("FIEuler", FIEuler(), 100f0),
]

println("FastIsostasy built-in stepper benchmark")
@printf("Domain: %d x %d points, tspan = %s, reltol = %g\n\n", 2^n, 2^n, TSPAN, RELTOL)
@printf("%-9s %10s %12s %11s %14s\n", "alg", "time [s]", "alloc [MiB]", "RHS evals", "peak |u| [m]")
println("-"^60)

for (name, alg, dtm) in algs
    r = measure(alg; reltol = RELTOL, dt_min = dtm)
    @printf("%-9s %10.4f %12.2f %11d %14.4f\n",
        name, r.time, r.bytes / 2^20, r.rhs, r.peak)
end

println("\n'RHS evals' counts calls to update_diagnostics! (from sim.timer).")
