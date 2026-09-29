# Kinematic BSL formalism of Adhikari et al. (2020), `AdhikariBSLFormalism`.
#
# The tests are driven at the level of `update_delta_V!` rather than through
# `run!`: every case of their Appendix A is a two-timestep, hand-computable
# scenario, and driving the increment directly is both sharper and cheaper than
# forcing a full simulation into the wanted regime. `step_delta_V!` reproduces
# exactly the call sequence of the sparse-diagnostics block, so what is exercised
# is the production path.
#
# Densities used below are the package defaults: ρ_i = 910, ρ_w = 1000,
# ρ_o = 1023 kg/m³, hence the floatation height H_0 = (ρ_o/ρ_i) max(S - B, 0)
# = 1.124176 (S - B).

using FastIsostasy
using Test

const RHO_I = 0.910e3
const RHO_W = 1.0e3
const RHO_O = 1.023e3
const SW_ICE = RHO_O / RHO_I            # 1.124176, the (ρ_o/ρ_i) of H_0

function build_bsl_sim(formalism; n = 3, W = 3.0e6, z_b = -500.0)
    domain = RegionalDomain(W, n)
    H0 = zeros(domain)
    it = TimeInterpolatedIceThickness([0.0, 1.0e4], [H0, H0], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = RigidLithosphere(), mantle = RigidMantle())
    opts = SolverOptions(; show_progress = false, integ = EulerIntegrator(dt = 10.0))
    sealevel = RegionalSeaLevel(formalism = formalism)
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    return Simulation(
        domain,
        bcs,
        sealevel,
        se,
        (0.0, 100.0);
        opts = opts,
        nout = nout,
        z_b_ref = fill(z_b, domain),
    )
end

# Put `sim.now` in the state `(H_ice, z_b, z_ss)` exactly as the sparse-diagnostics
# block does, then close the coupling interval and return its BSL contribution
# (m³ withheld from the ocean; positive means a falling BSL).
function step_delta_V!(sim, formalism, H_ice, z_b, z_ss)
    sim.now.H_ice .= H_ice
    sim.now.z_b .= z_b
    sim.now.z_ss .= z_ss
    FastIsostasy.update_Haf!(sim)
    FastIsostasy.update_maskgrounded!(sim)
    FastIsostasy.update_maskocean!(sim)
    FastIsostasy.update_HF!(sim)
    FastIsostasy.update_delta_V!(sim, formalism)
    return sim.now.delta_V
end

# Both formalisms need one call to prime what they difference against (the three
# absolute volumes for Goelzer, the two-time-level buffers for Adhikari), so a
# scenario is always "prime with the state at t, then measure at t + Δt".
function scenario_delta_V(formalism, state_t, state_tdt; kwargs...)
    sim = build_bsl_sim(formalism; kwargs...)
    step_delta_V!(sim, formalism, state_t...)
    return step_delta_V!(sim, formalism, state_tdt...)
end

adhikari_delta_V(args...; kwargs...) =
    scenario_delta_V(AdhikariBSLFormalism(), args...; kwargs...)
goelzer_delta_V(args...; kwargs...) =
    scenario_delta_V(GoelzerBSLFormalism(), args...; kwargs...)

@testset "Adhikari (2020) kinematic BSL formalism" begin

    # ---------------------------------------------------------------------
    # A1 — the signed height above floatation
    # ---------------------------------------------------------------------
    @testset "H_F (their Eqs. 7-8)" begin
        sim = build_bsl_sim(AdhikariBSLFormalism())
        z_b, z_ss = -500.0, 0.0
        H_0 = SW_ICE * (z_ss - z_b)

        # A grounded column: H_F is the ice in excess of the floatation height.
        step_delta_V!(sim, AdhikariBSLFormalism(), 700.0, z_b, z_ss)
        @test all(≈(700.0 - H_0), sim.now.H_F)
        @test all(sim.now.maskgrounded)

        # A floating column: H_F vanishes with the grounded mask 𝒢.
        step_delta_V!(sim, AdhikariBSLFormalism(), 500.0, z_b, z_ss)
        @test all(iszero, sim.now.H_F)
        @test !any(sim.now.maskgrounded)

        # With a pointwise ocean mask (no connectivity repair of their Eq. 3) a
        # cell is grounded only where H > H_0, so the signed H_F and the clamped
        # H_af coincide everywhere. This is the invariant the repair will break.
        for H in (0.0, 300.0, 562.0, 563.0, 1200.0)
            step_delta_V!(sim, AdhikariBSLFormalism(), H, z_b, z_ss)
            @test sim.now.H_F == sim.now.H_af
            @test all(>=(0), sim.now.H_F)
        end
    end

    # ---------------------------------------------------------------------
    # A5 — equivalence gate (their Appendix A, Case A1.1)
    # ---------------------------------------------------------------------
    # With ΔR = ΔS - ΔB = 0 the two formalisms are algebraically identical, in
    # every regime. Asserted as a strict equality (up to floating-point
    # roundoff), not a ratio: the freshwater-vs-seawater conversion cancels
    # between ΔH_V and Goelzer's V_den.
    @testset "equivalence with Goelzer at ΔR = 0" begin
        z_b, z_ss = -500.0, 0.0
        H_0 = SW_ICE * (z_ss - z_b)     # 562.09 m

        cases = [
            "regime 1, marine grounded" => (900.0, 800.0),
            "regime 1, ice advance" => (0.0, 900.0),
            "regime 2, grounding-line retreat" => (700.0, 500.0),
            "regime 2, grounding-line advance" => (500.0, 700.0),
            "regime 3, ice shelf thinning" => (500.0, 400.0),
            "regime 3, ice shelf thickening" => (400.0, 500.0),
        ]
        for (name, (H_t, H_tdt)) in cases
            adh = adhikari_delta_V((H_t, z_b, z_ss), (H_tdt, z_b, z_ss))
            goe = goelzer_delta_V((H_t, z_b, z_ss), (H_tdt, z_b, z_ss))
            @test isapprox(adh, goe, rtol = 1e-12) || error(name)
        end

        # Above sea level (H_0 = 0) the freshwater conversion is all there is.
        sim = build_bsl_sim(AdhikariBSLFormalism())
        A_tot = sum(sim.domain.A)
        adh = adhikari_delta_V((1000.0, 100.0, 0.0), (900.0, 100.0, 0.0))
        @test adh ≈ -100.0 * A_tot * RHO_I / RHO_W rtol = 1e-12
        @test H_0 > 0                   # the marine cases above really were marine
    end

    # ---------------------------------------------------------------------
    # Where the two formalisms genuinely differ: their Case A1.2 / Fig. A1a.
    # ---------------------------------------------------------------------
    # Grounded marine ice of unchanging thickness under a falling RSL. Adhikari
    # attributes no sea-level contribution to it (ΔH = 0 in Regime 1); the HAF
    # method sees H_af grow with the falling sea surface and wrongly reports a
    # BSL fall. This is the ~5 % mass-component difference of their Fig. 3e.
    @testset "Regime 1 with evolving RSL (their Case A1.2)" begin
        state_t = (700.0, -500.0, 0.0)
        state_tdt = (700.0, -500.0, -100.0)     # ΔR = -100 m, ΔH = 0
        adh = adhikari_delta_V(state_t, state_tdt)
        goe = goelzer_delta_V(state_t, state_tdt)
        @test adh == 0.0                        # no ice was exchanged
        @test goe > 0.0                         # HAF: a spurious BSL fall
    end

    # ---------------------------------------------------------------------
    # A5 — the six sign cases of their Appendix A2
    # ---------------------------------------------------------------------
    # `delta_V` is the volume withheld from the ocean, so `delta_V < 0` is a GMSL
    # rise and `delta_V > 0` a GMSL fall (their Eq. A3 with the opposite sign).
    @testset "sign cases (their Appendix A2)" begin
        z_b = -500.0

        # --- grounded → floating; H_F(t + Δt) = 0 ------------------------------
        # A2.1: ΔR ≤ 0 and ΔH < 0, with ΔH < ΔH_F < 0 ⇒ GMSL rise.
        @test adhikari_delta_V((700.0, z_b, 0.0), (500.0, z_b, 0.0)) < 0

        # A2.2: ΔR > 0 and ΔH = 0 ⇒ GMSL rise. The ice contributes to sea-level
        # rise although its thickness does not change — the case the HAF method
        # gets qualitatively wrong when it ignores the sea surface.
        @test adhikari_delta_V((600.0, z_b, 0.0), (600.0, z_b, 100.0)) < 0

        # A2.3: ΔR > 0 and ΔH ≠ 0. GMSL rises even when the ice thickens …
        @test adhikari_delta_V((600.0, z_b, 0.0), (610.0, z_b, 100.0)) < 0
        # … unless |ΔH| dominates |ΔH_F| by the factor ≈ 35 of their Eq. (A3),
        # here 100 m of thickening against 1 m of height above floatation.
        @test adhikari_delta_V((563.088, z_b, 0.0), (663.088, z_b, 100.0)) > 0

        # --- floating → grounded; H_F(t) = 0 ----------------------------------
        # A2.4: ΔR ≥ 0 and ΔH > 0, with 0 < ΔH_F < ΔH ⇒ GMSL fall.
        @test adhikari_delta_V((500.0, z_b, 0.0), (700.0, z_b, 0.0)) > 0

        # A2.5: ΔR < 0 and ΔH = 0 ⇒ GMSL fall. The mirror image of A2.2.
        @test adhikari_delta_V((500.0, z_b, 0.0), (500.0, z_b, -100.0)) > 0

        # A2.6: ΔR < 0 and ΔH ≠ 0. GMSL falls even when the ice thins …
        @test adhikari_delta_V((500.0, z_b, 0.0), (490.0, z_b, -100.0)) > 0
        # … unless |ΔH| dominates ΔH_F, here 499 m of thinning against 1 m.
        @test adhikari_delta_V((500.0, z_b, 0.0), (1.0, z_b, -500.0)) < 0
    end

    # ---------------------------------------------------------------------
    # A5 — floating ice only (their Appendix A3 / Fig. A1d)
    # ---------------------------------------------------------------------
    @testset "floating ice only (their Appendix A3)" begin
        sim = build_bsl_sim(AdhikariBSLFormalism())
        A_tot = sum(sim.domain.A)
        z_b, z_ss = -500.0, 0.0
        dH = -100.0                     # thinning shelf, floating at both times

        step_delta_V!(sim, AdhikariBSLFormalism(), 500.0, z_b, z_ss)
        adh = step_delta_V!(sim, AdhikariBSLFormalism(), 500.0 + dH, z_b, z_ss)

        # ΔH_M = 0: no ocean *mass* is exchanged, only volume.
        @test all(iszero, sim.now.kinematic.delta_H_M)
        # (1 - ρ_w/ρ_o) ΔH of freshwater in excess of the seawater displaced.
        @test adh ≈ (RHO_I / RHO_W) * (1 - RHO_W / RHO_O) * dH * A_tot rtol = 1e-12
        @test adh < 0                   # thinning shelf ⇒ GMSL rise

        # Goelzer's V_den is the same statement in the other decomposition.
        @test goelzer_delta_V((500.0, z_b, z_ss), (500.0 + dH, z_b, z_ss)) ≈ adh rtol =
            1e-12
    end

    # ---------------------------------------------------------------------
    # ΔH_M / ΔH_V split, and the buffers rolling forward
    # ---------------------------------------------------------------------
    @testset "increment buffers" begin
        sim = build_bsl_sim(AdhikariBSLFormalism())
        k = sim.now.kinematic
        @test FastIsostasy.kinematic_active(k)
        z_b = -500.0

        # Regime 3: everything is in the volume-only component.
        step_delta_V!(sim, AdhikariBSLFormalism(), 500.0, z_b, 0.0)
        step_delta_V!(sim, AdhikariBSLFormalism(), 400.0, z_b, 0.0)
        @test all(iszero, k.delta_H_M)
        @test all(<(0), k.delta_H_V)

        # Regime 1 above sea level: everything is in the mass component.
        step_delta_V!(sim, AdhikariBSLFormalism(), 1000.0, 100.0, 0.0)
        step_delta_V!(sim, AdhikariBSLFormalism(), 900.0, 100.0, 0.0)
        @test all(≈(-100.0), k.delta_H_M)
        @test all(iszero, k.delta_H_V)

        # The buffers hold the *end* of the interval just closed.
        @test all(≈(900.0), k.H_ice_prev)
        @test all(≈(900.0), k.H_F_prev)
        @test all(≈(1.0), k.maskland_prev)

        # A closed interval with no change contributes nothing.
        @test step_delta_V!(sim, AdhikariBSLFormalism(), 900.0, 100.0, 0.0) == 0.0
    end

    # `GoelzerBSLFormalism` must not pay for the two-time-level buffers.
    @testset "buffers are zero-size under GoelzerBSLFormalism" begin
        sim = build_bsl_sim(GoelzerBSLFormalism())
        k = sim.now.kinematic
        @test !FastIsostasy.kinematic_active(k)
        @test all(isempty, (k.H_ice_prev, k.H_F_prev, k.maskland_prev,
            k.delta_H_M, k.delta_H_V))
    end

    # ---------------------------------------------------------------------
    # End-to-end: a full forward run, and the state machinery around it
    # ---------------------------------------------------------------------
    @testset "forward run" begin
        domain = RegionalDomain(3.0e6, 5)
        H_grounded = 1.0e3 .* (domain.R .< 1.0e6)
        H_gone = zeros(domain)
        it = TimeInterpolatedIceThickness(
            [0.0, 1.0e3, 1.0e4],
            [H_grounded, H_gone, H_gone],
            domain,
        )
        bcs = BoundaryConditions(domain, ice_thickness = it)
        se = SolidEarth(domain; layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
        opts = SolverOptions(; show_progress = false, integ = EulerIntegrator(dt = 50.0))
        sealevel = RegionalSeaLevel(
            formalism = AdhikariBSLFormalism(),
            bsl = PiecewiseConstantBSL(),
        )
        nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
        sim = Simulation(
            domain,
            bcs,
            sealevel,
            se,
            (0.0, 1.0e3);
            opts = opts,
            nout = nout,
            z_b_ref = fill(-500.0, domain),
        )
        run!(sim)

        # The ice sheet melted away, so the BSL must have risen.
        @test sim.sealevel.bsl.z > 0
        @test sim.now.u != sim.ref.u                    # and it deformed the bedrock

        # `reset_state!` rewinds the two-time-level buffers along with the rest,
        # so a re-run reproduces the trajectory.
        z1, u1 = sim.sealevel.bsl.z, copy(sim.now.u)
        FastIsostasy.reset_state!(sim)
        sim.sealevel.bsl.z = sim.sealevel.bsl.ref.z
        sim.sealevel.bsl.A = sim.sealevel.bsl.ref.A
        # the buffers are primed from `sim.ref`, i.e. the state at t_span[1]
        @test sim.now.kinematic.H_ice_prev == sim.ref.H_ice
        @test sim.now.kinematic.H_F_prev == sim.ref.H_F
        run!(sim)
        @test sim.sealevel.bsl.z ≈ z1
        @test sim.now.u ≈ u1

        # `snapshot!`/`restore!` must copy the buffers, not alias them.
        buf = StateSnapshot(sim)
        snapshot!(buf, sim)
        @test buf.now.kinematic.H_ice_prev !== sim.now.kinematic.H_ice_prev
        @test buf.now.kinematic.H_ice_prev == sim.now.kinematic.H_ice_prev
        sim.now.kinematic.H_ice_prev .= -1.0
        @test !all(≈(-1.0), buf.now.kinematic.H_ice_prev)
        restore!(sim, buf)
        @test !all(≈(-1.0), sim.now.kinematic.H_ice_prev)
    end

    # The deprecated three-keyword path still builds the equivalent formalism.
    @testset "deprecated contribution keywords" begin
        sl = RegionalSeaLevel(
            volume_contribution = GoelzerVolumeContribution(),
            density_contribution = NoDensityContribution(),
            adjustment_contribution = GoelzerAdjustmentContribution(),
        )
        @test sl.formalism isa GoelzerBSLFormalism
        @test sl.formalism.volume isa GoelzerVolumeContribution
        @test sl.formalism.density isa NoDensityContribution
        @test sl.formalism.adjustment isa GoelzerAdjustmentContribution

        # Given both, the deprecated keywords win: an old script keeps doing what
        # it used to do.
        sl2 = RegionalSeaLevel(
            formalism = AdhikariBSLFormalism(),
            volume_contribution = NoVolumeContribution(),
        )
        @test sl2.formalism isa GoelzerBSLFormalism
        @test sl2.formalism.volume isa NoVolumeContribution
    end
end
