# Tests for the self-contained explicit RK integrators in src/integrators.jl.
# These check (a) that the adaptive solvers hit analytic solutions within the
# requested tolerance, (b) that each method achieves its nominal convergence
# order, and (c) that generic array shapes (vectors, matrices) work.

using FastIsostasy
using LinearAlgebra
using Test

# In-place fixed-step integration that bypasses the adaptive controller, so we
# can measure the pure convergence order of a tableau.
function fixed_step_solve(alg, f!, u0, tspan, N)
    dt = (tspan[2] - tspan[1]) / N
    integ = init_fi(f!, u0, tspan, alg; dt0 = dt)
    tab = integ.tableau
    for _ in 1:N
        FastIsostasy.perform_step!(integ, dt)
        integ.t += dt
        copyto!(integ.u, integ.unew)
        tab.fsal ? copyto!(integ.ks[1], integ.ks[end]) :
            integ.f!(integ.ks[1], integ.u, integ.p, integ.t)
    end
    return integ.u
end

# Empirical global-error order from two grid resolutions.
observed_order(errN, err2N) = log2(errN / err2N)

@testset "integrators" begin

    # Linear scalar decay: du/dt = -u, u(0) = 1  =>  u(t) = exp(-t).
    decay!(du, u, p, t) = (du .= -1.0 .* u; nothing)
    u_exact(t) = exp(-t)

    @testset "adaptive accuracy vs analytic" begin
        for alg in (FIBS3(), FITsit5())
            ts, us = fi_solve(decay!, [1.0], (0.0, 5.0), alg;
                reltol = 1e-8, abstol = 1e-10, saveat = 0.0:1.0:5.0)
            @test length(ts) == 6
            for (t, u) in zip(ts, us)
                @test isapprox(u[1], u_exact(t); atol = 1e-6)
            end
        end
    end

    @testset "tighter tolerance -> smaller error" begin
        err(reltol) = begin
            _, us = fi_solve(decay!, [1.0], (0.0, 5.0), FIBS3();
                reltol = reltol, abstol = reltol * 1e-2)
            abs(us[end][1] - u_exact(5.0))
        end
        @test err(1e-3) > err(1e-6) > err(1e-9)
    end

    @testset "convergence order" begin
        # Use a u-dependent, non-polynomial solution to exercise all stages.
        for (alg, nominal) in ((FIEuler(), 1), (FIBS3(), 3), (FITsit5(), 5))
            errs = Float64[]
            for N in (20, 40, 80)
                uN = fixed_step_solve(alg, decay!, [1.0], (0.0, 5.0), N)
                push!(errs, abs(uN[1] - u_exact(5.0)))
            end
            p1 = observed_order(errs[1], errs[2])
            p2 = observed_order(errs[2], errs[3])
            # Allow slack; Tsit5 can saturate against round-off, so only require
            # it to clearly exceed BS3's order.
            @test p1 > nominal - 0.6
            @test p2 > nominal - 0.6
        end
    end

    @testset "fixed-step Euler halves error when dt halves" begin
        e1 = abs(fi_solve(decay!, [1.0], (0.0, 5.0), FIEuler(); dt0 = 0.05)[2][end][1] - u_exact(5.0))
        e2 = abs(fi_solve(decay!, [1.0], (0.0, 5.0), FIEuler(); dt0 = 0.025)[2][end][1] - u_exact(5.0))
        @test isapprox(e1 / e2, 2.0; atol = 0.15)
    end

    @testset "vector system: harmonic oscillator" begin
        # u = [x, v], du = [v, -x]; x(0)=1, v(0)=0  =>  x(t)=cos t, v(t)=sin(-t).
        osc!(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
        ts, us = fi_solve(osc!, [1.0, 0.0], (0.0, 2pi), FITsit5();
            reltol = 1e-9, abstol = 1e-11, saveat = [pi, 2pi])
        @test isapprox(us[1], [cos(pi), -sin(pi)]; atol = 1e-6)   # t = pi
        @test isapprox(us[2], [cos(2pi), -sin(2pi)]; atol = 1e-6) # t = 2pi
    end

    @testset "matrix-shaped state" begin
        # Independent decay of every entry of a 2x2 matrix.
        rates = [-0.3 -0.7; -1.1 -1.9]
        matdecay!(du, u, p, t) = (@. du = rates * u; nothing)
        u0 = [1.0 2.0; 3.0 4.0]
        _, us = fi_solve(matdecay!, u0, (0.0, 2.0), FITsit5();
            reltol = 1e-9, abstol = 1e-11)
        @test isapprox(us[end], u0 .* exp.(rates .* 2.0); atol = 1e-6)
    end

    @testset "step statistics are sane" begin
        integ = init_fi(decay!, [1.0], (0.0, 5.0), FITsit5(); reltol = 1e-6)
        FastIsostasy.solve_to!(integ, 5.0, 10_000)
        @test integ.naccept > 0
        @test integ.nf >= integ.naccept   # at least one RHS eval per accepted step
        @test isapprox(integ.u[1], u_exact(5.0); atol = 1e-4)
    end

    # End-to-end `run!` on a small cylinder load: the three FIAlgorithms should
    # agree on the final viscous displacement field.
    @testset "run! backends agree on cylinder load" begin
        function build_sim(alg; reltol = 1f-5, dt_min = nothing)
            W, n = 3f6, 5
            domain = RegionalDomain(W, n)
            H0 = zeros(domain)
            H1 = 1f3 .* (domain.R .< 1f6)
            it = TimeInterpolatedIceThickness([0, 1, 10f3], [H0, H1, H1], domain)
            bcs = BoundaryConditions(domain, ice_thickness = it)
            se = SolidEarth(domain, layer_boundaries = [88f3],
                layer_viscosities = [1f21], rho_litho = 0f0)
            nout = NativeOutput(vars = [:u], t = [1000, 5000, 10_000f0])
            opts = SolverOptions(verbose = false,
                diffeq = DiffEqOptions(alg = alg, reltol = reltol, dt_min = dt_min))
            return Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, 10f3);
                nout = nout, opts = opts)
        end
        final_u(alg; kw...) = (s = build_sim(alg; kw...); run!(s); copy(s.now.u))

        ref = final_u(FIBS3())
        peak = maximum(abs, ref)
        @test peak > 1                                          # non-trivial response
        # Both adaptive methods should agree tightly at the same tolerance.
        @test maximum(abs, final_u(FITsit5()) .- ref) < 1f-3 * peak
        # Fixed-step Euler is cruder (first-order); allow ~0.5 % of the peak.
        @test maximum(abs, final_u(FIEuler(); dt_min = 100f0) .- ref) < 5f-3 * peak
    end
end
