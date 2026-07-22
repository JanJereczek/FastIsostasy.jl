# TransientCreepMantle (roadmaps/burgers.md Phases 1-2): the N = 1 Kelvin branch
# on the semi-implicit Crank-Nicolson path.
#
# The load-bearing checks are (a) the Δ → 0 limit reproduces `ViscousMantle` — the
# roadmap's permanent regression test, since the two schemes must agree term by
# term once the Kelvin branch locks — and (b) `update_dudt!` stays a *pure*
# function of `(u, t)` even though it advances Kelvin state, because the stepper
# calls it several times per step.

using FastIsostasy
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
    opts = SolverOptions(verbose = false, integ = EulerIntegrator(dt = dt))
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
        # Laterally variable parameters have no proven analogue (roadmap §8) ...
        s = build_creep_sim(burgers(1.2, 7.14); litho = LaterallyVariableLithosphere())
        @test_throws ErrorException run!(s)
        # ... and N > 1 is not wired into the solver yet.
        m3 = TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = (0.4, 0.4), kelvin_time = (1.0, 10.0))
        @test_throws ErrorException run!(build_creep_sim(m3))
    end
end
