# GPU counterpart of test_transient_creep.jl (roadmaps/burgers.md Phase 3: GPU
# validation). `TransientCreepMantle`'s state (`u_K`, 3D) and solver buffers
# (`PreAllocated.fftK`, 3D) were designed to be GPU-compatible from the start
# (roadmap §1, "State layout"), but this is the first time the general-N
# closed-form solve (`update_dudt!(..., ::TransientCreepMantle{MT,N}, ...)` in
# src/deformation.jl) actually runs on a `CuArray`: every per-branch buffer is a
# `view` into a 3D array and every update is a broadcast, so this exercises that
# GPU broadcasting over 3D-array views behaves the same as on CPU, for both
# N = 1 and N > 1.

using FastIsostasy, CUDA, Test

CUDA.allowscalar(false)

function build_gpu_creep_sim(mantle, backend; tend = 5f3, dt = 25f0, n = 5)
    W = 3f6
    domain = RegionalDomain(W, n; backend = backend)
    H0 = zeros(domain)
    H1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0f0, 1f0, tend], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyConstantLithosphere(), mantle = mantle,
        layer_boundaries = [88f3], layer_viscosities = [1f21])
    opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = dt))
    nout = NativeOutput(vars = [:u], t = [tend])
    return Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, tend);
        nout = nout, opts = opts)
end

burgers(Δ, τ) = TransientCreepMantle(
    shearmodulus = 67e9, relaxation_strength = Δ, kelvin_time = τ)

@testset "gpu TransientCreepMantle" begin
    @testset "N = 1: CPU and GPU agree" begin
        mantle = burgers(1.2, 7.14)
        sim_cpu = build_gpu_creep_sim(mantle, CPU())
        run!(sim_cpu)
        sim_gpu = build_gpu_creep_sim(mantle, CUDABackend())
        run!(sim_gpu)

        u_cpu, u_gpu = sim_cpu.now.u, Array(sim_gpu.now.u)
        uK_cpu, uK_gpu = sim_cpu.now.u_K, Array(sim_gpu.now.u_K)
        @test all(isfinite, u_gpu)
        @test all(isfinite, uK_gpu)
        peak = maximum(abs, u_cpu)
        @test peak > 1                                       # non-trivial response
        @test maximum(abs, u_gpu .- u_cpu) < 1f-2 * peak
        @test maximum(abs, uK_gpu .- uK_cpu) < 1f-2 * peak
    end

    @testset "N = 3: CPU and GPU agree" begin
        # Exercises the general-N branch loops (not just the N = 1 case above)
        # on GPU: per-branch views into the 3D `u_K`/`fftK` buffers, broadcast
        # over CuArray slices.
        mantle3 = TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = (0.4, 0.6, 0.3), kelvin_time = (1.0, 10.0, 100.0))
        sim_cpu = build_gpu_creep_sim(mantle3, CPU())
        run!(sim_cpu)
        sim_gpu = build_gpu_creep_sim(mantle3, CUDABackend())
        run!(sim_gpu)

        u_cpu, u_gpu = sim_cpu.now.u, Array(sim_gpu.now.u)
        uK_cpu, uK_gpu = sim_cpu.now.u_K, Array(sim_gpu.now.u_K)
        @test size(uK_gpu, 3) == 3
        @test all(isfinite, u_gpu)
        @test all(isfinite, uK_gpu)
        peak = maximum(abs, u_cpu)
        @test peak > 1
        @test maximum(abs, u_gpu .- u_cpu) < 1f-2 * peak
        @test maximum(abs, uK_gpu .- uK_cpu) < 1f-2 * peak
    end

    @testset "Δ → 0 reproduces ViscousMantle on GPU (regression)" begin
        sim_v = build_gpu_creep_sim(ViscousMantle(), CUDABackend())
        run!(sim_v)
        sim_locked = build_gpu_creep_sim(burgers(1e-9, 10.0), CUDABackend())
        run!(sim_locked)

        uv, u0 = Array(sim_v.now.u), Array(sim_locked.now.u)
        @test all(isfinite, uv)
        @test maximum(abs, u0 .- uv) < 1f-5 * maximum(abs, uv)
    end
end
