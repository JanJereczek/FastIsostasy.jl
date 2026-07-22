# TransientCreepMantle (roadmaps/burgers.md Phases 1-3): the N-branch Kelvin
# solve on the semi-implicit Crank-Nicolson path, plus Phase 3 validation
# (analytic disc-load solution, sanity checks).
#
# The load-bearing checks are (a) the Δ → 0 limit reproduces `ViscousMantle` — the
# roadmap's permanent regression test, since the two schemes must agree term by
# term once the Kelvin branch locks — (b) `update_dudt!` stays a *pure*
# function of `(u, t)` even though it advances Kelvin state, because the stepper
# calls it several times per step — and (c) the N = 1 solution matches the
# closed-form analytic disc-load solution derived in roadmap §2.4.

using FastIsostasy
using Random
using Statistics
using Test

const FI = FastIsostasy

function build_creep_sim(mantle; tend = 10f3, dt = 100f0, n = 6,
        litho = LaterallyConstantLithosphere())
    W = 3f6
    domain = RegionalDomain(W, n)
    H0 = zeros(domain)
    H1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0f0, 1f0, tend], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = litho, mantle = mantle,
        layer_boundaries = [88f3], layer_viscosities = [1f21])
    opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = dt))
    nout = NativeOutput(vars = [:u], t = [tend])
    return Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, tend);
        nout = nout, opts = opts)
end

final_u(mantle; kw...) = (s = build_creep_sim(mantle; kw...); run!(s); copy(s.now.u))

burgers(Δ, τ) = TransientCreepMantle(
    shearmodulus = 67e9, relaxation_strength = Δ, kelvin_time = τ)

@testset "TransientCreepMantle" begin

    @testset "construction" begin
        m = burgers(1.2, 7.14)
        @test FI.nbranches(m) == 1
        @test FI.nbranches(ViscousMantle()) == 0
        @test m isa TransientCreepMantle{Float64,1}
        # N branches: one Δⱼ per τⱼ, and the tuple length sets N
        m3 = TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = (0.4, 0.4, 0.4), kelvin_time = (1.0, 10.0, 100.0))
        @test FI.nbranches(m3) == 3
        @test_throws DimensionMismatch TransientCreepMantle(
            shearmodulus = 67e9, relaxation_strength = (1.0, 2.0), kelvin_time = 7.0)
        # Δ = 0 is the ViscousMantle limit, not a valid Kelvin branch
        @test_throws ArgumentError burgers(0.0, 7.14)
        @test_throws ArgumentError burgers(1.2, -1.0)
    end

    @testset "state is allocated per branch" begin
        s = build_creep_sim(burgers(1.2, 7.14))
        @test size(s.now.u_K, 3) == 1
        @test size(s.now.u_K_next) == size(s.now.u_K)
        sv = build_creep_sim(ViscousMantle())
        # steady creep carries no Kelvin state, so `u_K` costs nothing
        @test size(sv.now.u_K, 3) == 0
    end

    @testset "Δ → 0 reproduces ViscousMantle" begin
        # μ₂ = μ₁/Δ → ∞ locks the Kelvin branch (u_K → 0) and the coupled 2x2
        # solve collapses onto the ViscousMantle Crank-Nicolson update.
        uv = final_u(ViscousMantle())
        u0 = final_u(burgers(1e-9, 10.0))
        @test all(isfinite, uv)
        @test maximum(abs, u0 .- uv) < 1f-5 * maximum(abs, uv)
    end

    @testset "update_dudt! is pure at fixed t" begin
        # It advances Kelvin state, but the stepper calls it at init_problem!, at
        # FSAL priming and once per accepted step — so repeated calls at the same
        # `t` must not advance anything twice.
        s = build_creep_sim(burgers(1.2, 7.14))
        FI.init_problem!(s)
        d1, d2 = similar(s.now.u), similar(s.now.u)
        FI.update_diagnostics!(d1, s.now.u, s, 0f0)
        uK = copy(s.now.u_K)
        FI.update_diagnostics!(d2, s.now.u, s, 0f0)
        @test d1 == d2
        @test uK == s.now.u_K
    end

    @testset "transient enhances early subsidence, decaying with time" begin
        # The Ivins & Caron (2021) signature: a vigorous short-term enhancement
        # that relaxes back toward the steady-creep curve once t ≫ τ.
        run_to(mantle, tend) =
            (s = build_creep_sim(mantle; tend = tend, dt = 0.25f0); run!(s); s)

        sv_early, st_early = run_to(ViscousMantle(), 10f0), run_to(burgers(1.2, 7.14), 10f0)
        sv_late, st_late = run_to(ViscousMantle(), 50f0), run_to(burgers(1.2, 7.14), 50f0)
        c = (sv_early.domain.nx ÷ 2, sv_early.domain.ny ÷ 2)

        @test all(isfinite, st_late.now.u)
        @test any(!iszero, st_late.now.u_K)          # branch carries real state

        # Transient creep subsides further than steady creep at both times ...
        early = abs(st_early.now.u[c...]) / abs(sv_early.now.u[c...])
        late = abs(st_late.now.u[c...]) / abs(sv_late.now.u[c...])
        @test early > 1
        @test late > 1
        # ... but the excess shrinks once t grows past the retardation time τ.
        @test early > late
    end

    @testset "unsupported combinations error clearly" begin
        # Laterally variable parameters have no proven analogue (roadmap §8).
        s = build_creep_sim(burgers(1.2, 7.14); litho = LaterallyVariableLithosphere())
        @test_throws ErrorException run!(s)
    end

    @testset "N > 1 generalises the closed-form solve (roadmap §4)" begin
        # Two Kelvin branches with *identical* (Δ, τ) must give exactly the same
        # aggregate response as one branch with doubled relaxation strength: in
        # the Prony series, branches contribute additively to the compliance
        # (`u = u_M + Σⱼ u_K[j]`), so 2 branches of strength Δ ≡ 1 branch of
        # strength 2Δ. This cross-checks the general-N O(N) solve against the
        # independently-derived, already-validated N = 1 closed form.
        s1 = build_creep_sim(burgers(0.6, 7.14))
        s2 = build_creep_sim(TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = (0.3, 0.3), kelvin_time = (7.14, 7.14)))
        run!(s1); run!(s2)
        @test all(isfinite, s2.now.u)
        @test maximum(abs, s2.now.u .- s1.now.u) < 1f-5 * maximum(abs, s1.now.u)
        # By symmetry (identical parameters, identical zero initial condition),
        # the two branches must carry exactly the same state at every step.
        @test s2.now.u_K[:, :, 1] == s2.now.u_K[:, :, 2]

        # N = 3, distinct branches: shape, finiteness, and purity at fixed t.
        m3 = TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = (0.4, 0.6, 0.3), kelvin_time = (1.0, 10.0, 100.0))
        s3 = build_creep_sim(m3)
        @test size(s3.now.u_K, 3) == 3
        run!(s3)
        @test all(isfinite, s3.now.u)
        @test all(isfinite, s3.now.u_K)
        @test any(!iszero, s3.now.u_K)

        s3p = build_creep_sim(m3)
        FI.init_problem!(s3p)
        d1, d2 = similar(s3p.now.u), similar(s3p.now.u)
        FI.update_diagnostics!(d1, s3p.now.u, s3p, 0f0)
        uK = copy(s3p.now.u_K)
        FI.update_diagnostics!(d2, s3p.now.u, s3p, 0f0)
        @test d1 == d2
        @test uK == s3p.now.u_K

        # Δⱼ → 0 for every branch still reproduces ViscousMantle (permanent
        # regression, generalised from the N = 1 case above to N = 3).
        uv = final_u(ViscousMantle())
        m3_locked = TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = (1e-9, 1e-9, 1e-9), kelvin_time = (1.0, 10.0, 100.0))
        u0 = final_u(m3_locked)
        @test maximum(abs, u0 .- uv) < 1f-5 * maximum(abs, uv)
    end

    # --- Phase 3 validation (roadmap §5) -------------------------------------

    @testset "N = 1 matches the analytic disc-load solution (roadmap §5)" begin
        # Bueler et al. (2007) disc-load geometry, `rho_litho = 0` to match the
        # analytic derivation exactly (see docs/src/examples/benchmark_analytic.jl
        # and `analytic_solution`'s docstring) — same geometry as the historical
        # `benchmark1_compare` in test/old/test_benchmarks.jl, which accepted
        # `mean_error < 6, max_error < 7` for the *plain* `ViscousMantle` case
        # against this same analytic solution. That residual is dominated by the
        # finite periodic FFT domain vs. the analytic infinite half-space, not by
        # solver error: `ViscousMantle` alone shows the same-magnitude residual
        # with this identical harness (checked interactively while deriving this
        # test, not asserted here since it isn't this file's concern). So the
        # tolerance below matches the historical one, not an idealised zero.
        function build_disc_sim(mantle, dt; n = 6, tend = 2f3)
            W = 3f6
            domain = RegionalDomain(W, n)
            R0, H0 = 1f6, 1f3
            H_ice_0 = zeros(domain)
            H_ice_1 = H0 .* (domain.R .< R0)
            it = TimeInterpolatedIceThickness(Float32[0, 1, tend],
                [H_ice_0, H_ice_1, H_ice_1], domain)
            bcs = BoundaryConditions(domain, ice_thickness = it)
            se = SolidEarth(domain; lithosphere = LaterallyConstantLithosphere(),
                mantle = mantle, layer_boundaries = [88f3], layer_viscosities = [1f21],
                rho_litho = 0f0)
            opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = dt))
            nout = NativeOutput(vars = [:u], t = Float32.([50, 200, 500, 2000]))
            sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, tend);
                nout = nout, opts = opts)
            return sim, R0, H0
        end

        # Absolute errors along a transect, at every saved output time, against
        # `analytic_solution` — the same two-exponential closed form validated
        # against a direct RK4 integration of the 2-state ODE system while
        # deriving `src/analytic_solutions.jl` (see its comments for the
        # derivation and the sign convention of `relaxation_minus_1`).
        function transect_errors(sim, R0, H0)
            domain = sim.domain
            ii, jj = domain.mx:domain.nx, domain.my
            r = abs.(domain.X[ii, jj])
            errs = Float64[]
            for (k, t) in enumerate(sim.nout.t)
                u_num = sim.nout.vals[:u][k][ii, jj]
                u_ana = [FI.analytic_solution(Float64(rr), Float64(t) * sim.c.seconds_per_year,
                    sim.c, sim.solidearth, Float64(H0), Float64(R0)) for rr in r]
                append!(errs, abs.(Float64.(u_num) .- u_ana))
            end
            return errs
        end

        mantle = burgers(1.2, 7.14)

        sim, R0, H0 = build_disc_sim(mantle, 5f0)
        run!(sim)
        errs = transect_errors(sim, R0, H0)
        @test maximum(errs) < 7.0
        @test mean(errs) < 6.0

        # Convergence: halving-ish the step must not make the (dt-independent,
        # domain-size-dominated) error worse — this is the CN discretisation
        # error shrinking on top of that floor.
        sim_coarse, = build_disc_sim(mantle, 25f0)
        run!(sim_coarse)
        errs_coarse = transect_errors(sim_coarse, R0, H0)
        @test maximum(errs) <= maximum(errs_coarse)
        @test mean(errs) <= mean(errs_coarse)
    end

    @testset "sanity checks (roadmap §5)" begin
        # u_K stays bounded: it must never exceed the total displacement it's
        # part of (u = u_M + Σⱼ u_K[j]), and u_K → 0 at equilibrium (verified
        # analytically in src/analytic_solutions.jl) means it should never even
        # approach that bound for a physically reasonable Δ.
        @testset "u_K bounded" begin
            s = build_creep_sim(burgers(1.2, 7.14); tend = 5f3, dt = 25f0)
            run!(s)
            @test all(isfinite, s.now.u_K)
            @test maximum(abs, s.now.u_K) < maximum(abs, s.now.u)
        end

        # Monotone approach to equilibrium for a load that's held constant after
        # a short ramp: no overshoot/oscillation is expected, since the per-mode
        # CN system has real, negative eigenvalues only (roadmap §2.4 — the
        # discriminant of the quadratic in `relaxation_minus_1` is positive, so
        # both roots are real; both are also negative since the system is
        # dissipative). Checked at the disc centre, where the response is
        # smoothest.
        @testset "monotone approach to equilibrium" begin
            n_out = 40
            tend, dt = 5f3, 25f0
            W, n = 3f6, 6
            domain = RegionalDomain(W, n)
            H0 = zeros(domain)
            H1 = 1f3 .* (domain.R .< 1f6)
            it = TimeInterpolatedIceThickness([0f0, 1f0, tend], [H0, H1, H1], domain)
            bcs = BoundaryConditions(domain, ice_thickness = it)
            se = SolidEarth(domain; lithosphere = LaterallyConstantLithosphere(),
                mantle = burgers(1.2, 7.14), layer_boundaries = [88f3],
                layer_viscosities = [1f21])
            opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = dt))
            nout = NativeOutput(vars = [:u], t = collect(range(100f0, tend, length = n_out)))
            s = Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, tend);
                nout = nout, opts = opts)
            run!(s)
            mx, my = s.domain.mx, s.domain.my
            u_center = [abs(s.nout.vals[:u][k][mx, my]) for k in 1:n_out]
            @test all(isfinite, u_center)
            @test issorted(u_center)  # monotonically increasing toward equilibrium
        end

        # No spectral ringing at high k: a spatially rough load (strong
        # high-wavenumber content, the regime where CN's A-stable-but-not-
        # L-stable amplification factor is furthest from 0) must still relax
        # smoothly — bounded, finite, and with monotonically shrinking steps in
        # the tail of the run, not a growing or sign-alternating oscillation.
        @testset "no spectral ringing at high k" begin
            Random.seed!(42)
            W, n = 3f6, 5
            domain = RegionalDomain(W, n)
            H0 = zeros(domain)
            H1 = Float32.(500.0 .+ 400.0 .* (2 .* rand(domain.nx, domain.ny) .- 1))
            it = TimeInterpolatedIceThickness(Float32[0, 1, 1f4], [H0, H1, H1], domain)
            bcs = BoundaryConditions(domain, ice_thickness = it)
            se = SolidEarth(domain; lithosphere = LaterallyConstantLithosphere(),
                mantle = burgers(1.2, 7.14), layer_boundaries = [88f3],
                layer_viscosities = [1f21])
            # deliberately large dt relative to the Kelvin time (7.14 yr), to
            # stress-test the CN amplification factor's high-k behaviour.
            opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = 50f0))
            nout = NativeOutput(vars = [:u], t = collect(100f0:100f0:5f3))
            sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0f0, 5f3);
                nout = nout, opts = opts)
            run!(sim)

            @test all(isfinite, sim.now.u_K)
            maxabs_u = [maximum(abs, sim.nout.vals[:u][k]) for k in eachindex(sim.nout.t)]
            @test all(isfinite, maxabs_u)
            # Shrinking, same-sign increments in the tail: smooth relaxation, not
            # a growing or sign-alternating (ringing) oscillation.
            tail_increments = diff(maxabs_u)[(end-9):end]
            @test all(>=(0), tail_increments)
            @test issorted(tail_increments, rev = true)
        end
    end
end
