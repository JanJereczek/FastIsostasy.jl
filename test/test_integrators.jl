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
    integ = init_integrator(f!, u0, tspan, alg)
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
        for alg in (BS3Integrator{Float64}(reltol = 1e-8, abstol = 1e-10),
                Tsit5Integrator{Float64}(reltol = 1e-8, abstol = 1e-10))
            ts, us = integrate(decay!, [1.0], (0.0, 5.0), alg;
                saveat = 0.0:1.0:5.0)
            @test length(ts) == 6
            for (t, u) in zip(ts, us)
                @test isapprox(u[1], u_exact(t); atol = 1e-6)
            end
        end
    end

    @testset "tighter tolerance -> smaller error" begin
        err(reltol) = begin
            _, us = integrate(decay!, [1.0], (0.0, 5.0),
                BS3Integrator{Float64}(reltol = reltol, abstol = reltol * 1e-2))
            abs(us[end][1] - u_exact(5.0))
        end
        @test err(1e-3) > err(1e-6) > err(1e-9)
    end

    @testset "convergence order" begin
        # Use a u-dependent, non-polynomial solution to exercise all stages.
        for (alg, nominal) in
            ((EulerIntegrator(dt = 0.1), 1), (BS3Integrator(), 3), (Tsit5Integrator(), 5))
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
        e1 = abs(integrate(decay!, [1.0], (0.0, 5.0), EulerIntegrator(dt = 0.05))[2][end][1] - u_exact(5.0))
        e2 = abs(integrate(decay!, [1.0], (0.0, 5.0), EulerIntegrator(dt = 0.025))[2][end][1] - u_exact(5.0))
        @test isapprox(e1 / e2, 2.0; atol = 0.15)
    end

    @testset "vector system: harmonic oscillator" begin
        # u = [x, v], du = [v, -x]; x(0)=1, v(0)=0  =>  x(t)=cos t, v(t)=sin(-t).
        osc!(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
        ts, us = integrate(osc!, [1.0, 0.0], (0.0, 2pi),
            Tsit5Integrator{Float64}(reltol = 1e-9, abstol = 1e-11); saveat = [pi, 2pi])
        @test isapprox(us[1], [cos(pi), -sin(pi)]; atol = 1e-6)   # t = pi
        @test isapprox(us[2], [cos(2pi), -sin(2pi)]; atol = 1e-6) # t = 2pi
    end

    @testset "matrix-shaped state" begin
        # Independent decay of every entry of a 2x2 matrix.
        rates = [-0.3 -0.7; -1.1 -1.9]
        matdecay!(du, u, p, t) = (@. du = rates * u; nothing)
        u0 = [1.0 2.0; 3.0 4.0]
        _, us = integrate(matdecay!, u0, (0.0, 2.0),
            Tsit5Integrator{Float64}(reltol = 1e-9, abstol = 1e-11))
        @test isapprox(us[end], u0 .* exp.(rates .* 2.0); atol = 1e-6)
    end

    @testset "step statistics are sane" begin
        integ = init_integrator(decay!, [1.0], (0.0, 5.0), Tsit5Integrator{Float64}(reltol = 1e-6))
        FastIsostasy.solve_to!(integ, 5.0, 10_000)
        @test integ.naccept > 0
        @test integ.nf >= integ.naccept   # at least one RHS eval per accepted step
        @test isapprox(integ.u[1], u_exact(5.0); atol = 1e-4)
    end

    # End-to-end `run!` on a small cylinder load: the three AbstractIntegrators should
    # agree on the final viscous displacement field.
    @testset "run! backends agree on cylinder load" begin
        function build_sim(alg)
            W, n = 3f6, 5
            domain = RegionalDomain(W, n)
            H0 = zeros(domain)
            H1 = 1f3 .* (domain.R .< 1f6)
            it = TimeInterpolatedIceThickness([0, 1, 10f3], [H0, H1, H1], domain)
            bcs = BoundaryConditions(domain, ice_thickness = it)
            se = SolidEarth(domain, layer_boundaries = [88f3],
                layer_viscosities = [1f21], rho_litho = 0f0)
            nout = NativeOutput(vars = [:u], t = [1000, 5000, 10_000f0])
            opts = SolverOptions(verbose = false, integ = alg)
            return Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, 10f3);
                nout = nout, opts = opts)
        end
        final_u(alg) = (s = build_sim(alg); run!(s); copy(s.now.u))

        ref = final_u(BS3Integrator(reltol = 1f-5))
        peak = maximum(abs, ref)
        @test peak > 1                                          # non-trivial response
        # Both adaptive methods should agree tightly at the same tolerance.
        @test maximum(abs, final_u(Tsit5Integrator(reltol = 1f-5)) .- ref) < 1f-3 * peak
        # Fixed-step Euler is cruder (first-order); allow ~0.5 % of the peak.
        @test maximum(abs, final_u(EulerIntegrator(dt = 100f0)) .- ref) < 5f-3 * peak
        # RKCIntegrator uses SSV's own embedded estimate, so `reltol` means the same
        # thing as it does for Tsit5Integrator/BS3Integrator: it must agree just as tightly at
        # the shared default tolerance, with no special-cased reltol. (It used
        # to need reltol=1f-7 here to reach even 1 % — see the estimator note in
        # `perform_step!` for why that indicator was replaced.)
        @test maximum(abs, final_u(RKCIntegrator(reltol = 1f-5)) .- ref) < 1f-3 * peak
    end
end

# Tests for `RKCIntegrator` (roadmap stabilise_dt.md, Phase 2): the stabilised
# Runge-Kutta-Chebyshev stepper. Its coefficient formulas were sourced and
# cross-checked against SUNDIALS' LSRKStep (`arkode_lsrkstep.c`) after two
# independent from-memory reconstructions were empirically falsified — see the
# roadmap's Phase 2 notes. These tests exist specifically to catch a
# regression back to either of those falsified variants.
@testset "RKCIntegrator" begin
    decay!(du, u, p, t) = (du .= -1.0 .* u; nothing)
    u_exact(t) = exp(-t)

    @testset "recurrence is second order (Taylor coefficients on u'=λu)" begin
        # Y_s/Y_0 must equal 1 + z + z²/2 + O(z³) for every stage count and
        # damping tested — the defining property of RKC2, and the property
        # both falsified reconstructions failed (one gave order 1, the other
        # failed even that).
        for (s, damping) in ((2, 2/13), (5, 2/13), (13, 2/13), (37, 2/13), (13, 0.3))
            mu, nu, mutilde, gammatilde, _ = FastIsostasy.rkc_coeffs(Float64, s, damping)
            function Ys_over_Y0(z)
                F0 = z; y0 = 1.0
                y1 = y0 + mutilde[1] * F0
                y2 = y0
                for j in 2:s
                    Fjm1 = z * y1
                    ynext = mu[j]*y1 + nu[j]*y2 + (1-mu[j]-nu[j])*y0 +
                        mutilde[j]*Fjm1 + gammatilde[j]*F0
                    y2, y1 = y1, ynext
                end
                return y1
            end
            h = 1e-5
            p0, pp, pm = Ys_over_Y0(0.0), Ys_over_Y0(h), Ys_over_Y0(-h)
            @test isapprox(p0, 1.0; atol = 1e-10)
            @test isapprox((pp - pm) / (2h), 1.0; atol = 1e-4)         # p'(0) = 1
            @test isapprox((pp - 2p0 + pm) / h^2, 1.0; atol = 1e-2)    # p''(0) = 1
        end
    end

    @testset "stability boundary scales as ~0.653 s² (SSV default damping)" begin
        # Literature/roadmap value (§1/§2.2); the falsified 1st-order-only
        # reconstruction gave ~1.82 s² instead (larger boundary, wrong method).
        for s in (25, 50, 100, 200)
            beta = FastIsostasy.rkc_stability_boundary(s, 2/13)
            @test isapprox(beta / s^2, 0.653; atol = 0.03)
        end
        # Monotone in s (larger stage budget -> larger stability interval).
        betas = [FastIsostasy.rkc_stability_boundary(s, 2/13) for s in (5, 10, 25, 50)]
        @test issorted(betas)
    end

    @testset "damping <= 0 is rejected (w1's closed form is singular at w0=1)" begin
        @test_throws ArgumentError FastIsostasy.rkc_coeffs(Float64, 10, 0.0)
        @test_throws ArgumentError FastIsostasy.rkc_coeffs(Float64, 10, -0.1)
    end

    @testset "adaptive accuracy vs analytic" begin
        ts, us = integrate(decay!, [1.0], (0.0, 5.0),
            RKCIntegrator{Float64}(reltol = 1e-9, abstol = 1e-11); saveat = 0.0:1.0:5.0)
        @test length(ts) == 6
        for (t, u) in zip(ts, us)
            @test isapprox(u[1], u_exact(t); atol = 1e-6)
        end
    end

    @testset "tighter tolerance -> smaller error" begin
        err(reltol) = begin
            _, us = integrate(decay!, [1.0], (0.0, 5.0),
                RKCIntegrator{Float64}(reltol = reltol, abstol = reltol * 1e-2))
            abs(us[end][1] - u_exact(5.0))
        end
        @test err(1e-3) > err(1e-6) > err(1e-9)
    end

    @testset "fixed-step convergence order is 2" begin
        function fixed_step_solve_rkc(f!, u0, tspan, N)
            dt = (tspan[2] - tspan[1]) / N
            integ = init_integrator(f!, u0, tspan, RKCIntegrator{Float64}(dt0 = dt))
            for _ in 1:N
                FastIsostasy.perform_step!(integ, dt)
                integ.t += dt
                copyto!(integ.u, integ.unew)
            end
            return integ.u
        end
        errs = Float64[]
        for N in (20, 40, 80)
            uN = fixed_step_solve_rkc(decay!, [1.0], (0.0, 5.0), N)
            push!(errs, abs(uN[1] - u_exact(5.0)))
        end
        @test observed_order(errs[1], errs[2]) > 1.4   # nominal 2, allow slack
        @test observed_order(errs[2], errs[3]) > 1.4
    end

    @testset "steplog widens to (t, dt, s)" begin
        integ = init_integrator(decay!, [1.0], (0.0, 5.0), RKCIntegrator{Float64}(reltol = 1e-6))
        steplog = Tuple[]
        FastIsostasy.solve_to!(integ, 5.0, 10_000, steplog)
        @test !isempty(steplog)
        @test all(length(entry) == 3 for entry in steplog)
        @test all(entry[3] >= 2 for entry in steplog)   # stage count, always >= 2
    end

    @testset "stiff decay: far fewer RHS evaluations than Tsit5Integrator" begin
        # The core value proposition of RKC2 (roadmap §2.2): cost scales with
        # sqrt(stiffness) instead of stiffness. lambda chosen well beyond
        # Tsit5Integrator's stability-limited micro-stepping regime.
        lambda = -5e4
        stiffdecay!(du, u, p, t) = (du .= lambda .* u; nothing)
        u0 = [1.0]

        integ_rkc = init_integrator(stiffdecay!, u0, (0.0, 1.0),
            RKCIntegrator{Float64}(reltol = 1e-5, abstol = 1e-7))
        FastIsostasy.solve_to!(integ_rkc, 1.0, 10_000_000)

        integ_tsit = init_integrator(stiffdecay!, u0, (0.0, 1.0),
            Tsit5Integrator{Float64}(reltol = 1e-5, abstol = 1e-7))
        FastIsostasy.solve_to!(integ_tsit, 1.0, 10_000_000)

        @test integ_rkc.nf * 5 < integ_tsit.nf   # at least 5x fewer RHS evals
        @test isapprox(integ_rkc.u[1], 0.0; atol = 1e-6)
    end

    @testset "spectral-radius probe does not corrupt sim.now.u at init (regression)" begin
        # Regression test: RKCIntegrator's init_integrator used to run the spectral-radius power
        # iteration directly on `update_diagnostics!`, which writes every trial
        # state into `sim.now.u` (via `update_bedrock!`) — the very array object
        # passed in as `u0`. Without routing through `snapshotting_probe` (the
        # roadmap stabilise_dt.md §3 fix, previously wired only into
        # `simulation_rhs_probe`/`stiffness_report`, not into RKCIntegrator's own
        # estimator), `sim.now.u` — and hence the integrator's initial condition —
        # was left at the last probe trial instead of the true initial condition.
        W, n = 3.0e6, 5
        domain = RegionalDomain(W, n)
        H0 = zeros(domain)
        H1 = 1.0e3 .* (domain.R .< 1.0e6)
        it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
        bcs = BoundaryConditions(domain, ice_thickness = it)
        se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
            layer_boundaries = [88.0e3], layer_viscosities = [1.0e19])
        opts = SolverOptions(verbose = false, integ = RKCIntegrator())
        nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
        sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, 100.0);
            opts = opts, nout = nout)
        FastIsostasy.init_problem!(sim)
        u_before = copy(sim.now.u)
        n_before = sim.now.count_sparse_updates

        integ = FastIsostasy.build_integrator(sim)

        @test sim.now.u == u_before
        @test sim.now.count_sparse_updates == n_before
        @test integ.u == u_before
    end
end
