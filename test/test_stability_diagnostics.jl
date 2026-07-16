# Tests for src/stability.jl (roadmap: stabilise_dt.md, Phase 0): the explicit-RK
# stability-boundary calculator and the nonlinear-power-iteration spectral-radius
# estimator, on both toy ODEs (closed-form answers) and a real LV-Maxwell
# `Simulation` (sanity + the no-side-effects guarantee `simulation_rhs_probe` exists
# to provide).

using FastIsostasy
using Test

@testset "stability diagnostics" begin

    @testset "real-axis stability limits" begin
        lim_euler = stability_limit(FIEuler())
        lim_bs3 = stability_limit(FIBS3())
        lim_tsit5 = stability_limit(FITsit5())

        # Known values (Euler: exactly 2; BS3/Tsit5: standard embedded-pair limits).
        @test isapprox(lim_euler, 2.0; atol = 1e-4)
        @test 2.4 < lim_bs3 < 2.6
        @test 3.4 < lim_tsit5 < 3.6
        @test lim_euler < lim_bs3 < lim_tsit5   # higher order -> larger stability interval
    end

    @testset "spectral radius on toy ODEs" begin
        # du/dt = -u: Jacobian is the scalar -1, spectral radius exactly 1.
        decay!(du, u, p, t) = (du .= -1.0 .* u; nothing)
        lam = spectral_radius_estimate(decay!, [1.0], nothing, 0.0; tol = 1e-10)
        @test isapprox(lam, 1.0; atol = 1e-6)

        # Elementwise (diagonal-Jacobian) decay: spectral radius = max(abs.(rates)).
        rates = [-0.3 -0.7; -1.1 -1.9]
        matdecay!(du, u, p, t) = (@. du = rates * u; nothing)
        lam2 = spectral_radius_estimate(matdecay!, [1.0 2.0; 3.0 4.0], nothing, 0.0; tol = 1e-10)
        @test isapprox(lam2, maximum(abs, rates); atol = 1e-3)

        # Loose vs tight tolerance must agree once actually converged (regression
        # test for the state-corruption bug where repeated evaluations on a shared,
        # mutated buffer produced tolerance-dependent, non-reproducible answers).
        lam_loose = spectral_radius_estimate(matdecay!, [1.0 2.0; 3.0 4.0], nothing, 0.0;
            maxiter = 100, tol = 1e-2)
        lam_tight = spectral_radius_estimate(matdecay!, [1.0 2.0; 3.0 4.0], nothing, 0.0;
            maxiter = 500, tol = 1e-10)
        @test isapprox(lam_loose, lam_tight; rtol = 1e-2)
    end

    @testset "degenerate (all-zero RHS) fallback avoids the BC null space" begin
        # A RHS that is identically zero at u0 forces the power iteration onto its
        # fallback direction. A spatially uniform fallback would be silently
        # annihilated by any mean-subtracting BC and converge to a spurious 0; the
        # alternating-sign fallback must not.
        zero_then_scale!(du, u, p, t) = (@. du = -u; nothing)  # zero at u0=0, but Jacobian is -I
        lam = spectral_radius_estimate(zero_then_scale!, zeros(4, 4), nothing, 0.0; tol = 1e-8)
        @test isapprox(lam, 1.0; atol = 1e-4)
    end

    function build_lv_sim(; n = 5, W = 3.0e6, layer_viscosities = [1e19, 1e21])
        domain = RegionalDomain(W, n)
        H0 = zeros(domain)
        H1 = 1.0e3 .* (domain.R .< 1.0e6)
        it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
        bcs = BoundaryConditions(domain, ice_thickness = it)
        se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
            layer_boundaries = [88.0e3, 400e3], layer_viscosities = layer_viscosities)
        opts = SolverOptions(; verbose = false,
            diffeq = DiffEqOptions(alg = FITsit5(), dt_min = 1.0))
        nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
        sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, 100.0);
            opts = opts, nout = nout)
        FastIsostasy.init_problem!(sim)
        return sim
    end

    @testset "simulation_rhs_probe leaves the simulation untouched" begin
        sim = build_lv_sim()
        u_before = copy(sim.now.u)
        ue_before = copy(sim.now.ue)
        load_before = copy(sim.now.columnanoms.load)
        n_before = sim.now.count_sparse_updates
        t_before = sim.timer.t

        stiffness_report(sim; t = 10.0, maxiter = 30)

        @test sim.now.u == u_before
        @test sim.now.ue == ue_before
        @test sim.now.columnanoms.load == load_before
        @test sim.now.count_sparse_updates == n_before
        @test sim.timer.t == t_before
    end

    @testset "stiffness_report: finiteness and physical sensitivity" begin
        sim = build_lv_sim(layer_viscosities = [1e19, 1e21])
        rep = stiffness_report(sim; t = 10.0)
        @test isfinite(rep.lambda_max) && rep.lambda_max > 0
        @test rep.dt_euler < rep.dt_bs3 < rep.dt_tsit5   # larger stability interval -> larger dt
        @test all(isfinite, (rep.analytic_bound.dc, rep.analytic_bound.k_max))

        # Lowering the channel viscosity must increase the estimated stiffness
        # (lambda ~ 1/eta in the U-shaped bound, roadmaps/stabilise_dt.md §2.1).
        sim_stiffer = build_lv_sim(layer_viscosities = [1e18, 1e21])
        rep_stiffer = stiffness_report(sim_stiffer; t = 10.0)
        @test rep_stiffer.lambda_max > rep.lambda_max
    end
end
