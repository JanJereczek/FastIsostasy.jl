# GPU counterpart of test_integrators.jl's "run! backends agree on cylinder
# load" test, focused on `RKCIntegrator` (roadmap stabilise_dt.md, Phase 3: GPU
# validation). Everything `RKCIntegrator` needs — the three-term Chebyshev recurrence,
# the spectral-radius power iteration, and the PI step-size controller's
# `norm`-based error estimate — is written as generic broadcasts/reductions, so
# this is the first exercise of any *adaptive* `AbstractIntegrator` on `CuArray`
# (previously only `EulerIntegrator`'s fixed-step path had GPU coverage, via the AD
# validity tests in `test_ad_validity_gpu.jl`).

using FastIsostasy, CUDA, Test

CUDA.allowscalar(false)

function build_gpu_sim(alg, arraykernel; reltol = 1f-5, dt_min = nothing)
    W, n = 3f6, 5
    domain = RegionalDomain(W, n; arraykernel = arraykernel)
    H0 = zeros(domain)
    H1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0f0, 1f0, 10f3], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain, layer_boundaries = [88f3],
        layer_viscosities = [1f21], rho_litho = 0f0)
    nout = NativeOutput(vars = [:u], t = [1000f0, 5000f0, 10_000f0])
    opts = SolverOptions(verbose = false,
        diffeq = DiffEqOptions(alg = alg, reltol = reltol, dt_min = dt_min))
    return Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, 10f3);
        nout = nout, opts = opts)
end

@testset "gpu RKCIntegrator" begin
    @testset "run! completes on GPU and matches CPU within tolerance" begin
        # Both run at the same reltol: RKCIntegrator uses SSV's own embedded estimate,
        # so its `reltol` is calibrated like BS3Integrator's/Tsit5Integrator's — mirrors the CPU
        # comparison in test_integrators.jl.
        for (name, alg, reltol) in (("RKCIntegrator", RKCIntegrator(), 1f-5), ("BS3Integrator", BS3Integrator(), 1f-5))
            sim_cpu = build_gpu_sim(alg, Array; reltol = reltol)
            run!(sim_cpu)
            sim_gpu = build_gpu_sim(alg, CuArray; reltol = reltol)
            run!(sim_gpu)

            u_cpu = sim_cpu.now.u
            u_gpu = Array(sim_gpu.now.u)
            @test all(isfinite, u_gpu)
            peak = maximum(abs, u_cpu)
            @test peak > 1                                      # non-trivial response
            @test maximum(abs, u_gpu .- u_cpu) < 1f-2 * peak
        end
    end

    @testset "spectral-radius probe does not corrupt sim.now.u on GPU (regression)" begin
        # GPU counterpart of the CPU regression test in test_integrators.jl:
        # RKCIntegrator's init-time spectral-radius estimate must leave `sim.now.u`
        # (the very array passed in as `u0`) untouched, on CuArray too.
        sim = build_gpu_sim(RKCIntegrator(), CuArray; reltol = 1f-5)
        FastIsostasy.init_problem!(sim)
        u_before = Array(sim.now.u)

        integ = FastIsostasy.build_integrator(sim)

        @test isfinite(integ.lambda_max) && integ.lambda_max > 0
        @test Array(sim.now.u) == u_before
        @test Array(integ.u) == u_before
    end
end
