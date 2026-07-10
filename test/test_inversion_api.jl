# Tests for the (AD-free) inversion API in src/inverse/: encodings, observations,
# regularizations, and the `loss` pipeline (reconstruct! → forward → misfit).

using FastIsostasy
using Test

function build_test_sim(; n = 5, tend = 5f3)
    W = 3f6
    domain = RegionalDomain(W, n)
    H0 = zeros(domain)
    H1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0f0, 1f0, 50f3], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88f3], layer_viscosities = [1f21])
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0, tend))
    return sim, se
end

@testset "inversion API" begin

    @testset "encodings: nparams & reconstruct!" begin
        sim, se = build_test_sim()
        enc = Test2Encoding()
        @test nparams(enc) == 19
        θ = Float32[21.0, -1f6,0f0,8f5,0.5, 1f6,0f0,8f5,-0.5, 0f0,0f0,8f5,0f0, 0f0,0f0,8f5,0f0,
            3200f0, 2600f0]
        reconstruct!(sim, θ, enc)
        @test sim.solidearth.rho_uppermantle == 3200f0
        @test sim.solidearth.rho_litho == 2600f0
        # +0.5 decade bump at (-1e6,0) and -0.5 at (1e6,0) => field brackets 10^21
        lv = log10.(sim.solidearth.effective_viscosity)
        @test maximum(lv) > 21 && minimum(lv) < 21

        enc1 = Test1Encoding(Float32.(collect(0:4)), (1f6, 1f6, 1f6), (-1f0, 1f0))
        @test nparams(enc1) == 28
    end

    @testset "Observation construction" begin
        pts = [CartesianIndex(3, 3), CartesianIndex(4, 4)]
        obs = Observation(VerticalUpliftObservable(), pts, Float32[1f3, 2f3],
            Float32[1, 2, 3, 4]; σ = 0.5f0)
        @test FastIsostasy.nentries(obs) == 4
        @test_throws DimensionMismatch Observation(VerticalUpliftObservable(), pts,
            Float32[1f3], Float32[1, 2, 3])            # 2 pts * 1 time != 3
    end

    @testset "diff-mode encoding requirement" begin
        sim, _ = build_test_sim()
        obs = Observation(VerticalUpliftObservable(), [CartesianIndex(3, 3)],
            Float32[5f3], Float32[0])
        # TangentMode requires an encoding
        @test_throws ArgumentError ParameterInversion(sim, nothing, [obs];
            diffmode = TangentMode())
        # AdjointMode does not
        @test ParameterInversion(sim, nothing, [obs]; diffmode = AdjointMode()) isa
            ParameterInversion
    end

    @testset "loss self-consistency & sensitivity" begin
        sim, se = build_test_sim()
        enc = Test2Encoding()
        θ_true = Float32[21.0, -1f6,-1f6,8f5,0.3, 1f6,1f6,8f5,-0.3,
            -1f6,1f6,8f5,0.2, 1f6,-1f6,8f5,-0.2, se.rho_uppermantle, se.rho_litho]
        pts = [CartesianIndex(i, j) for i in 14:18 for j in 15:17][1:8]

        # synthetic ground truth
        reconstruct!(sim, θ_true, enc)
        FastIsostasy.reset_state!(sim)
        FastIsostasy.init_problem!(sim)
        integ = FastIsostasy.build_integrator(sim)
        FastIsostasy.advance_with_output!(integ, sim, 5f3)
        data = Float32[sim.now.u[p] + sim.now.ue[p] for p in pts]
        @test any(<(−1f0), data)          # nonzero subsidence signal

        obs = Observation(VerticalUpliftObservable(), pts, Float32[5f3], data; σ = 0.1f0)
        prob = ParameterInversion(sim, enc, [obs])

        @test loss(prob, θ_true) < 1f-2   # ≈ 0 at truth
        θbad = copy(θ_true); θbad[1] += 0.2f0
        @test loss(prob, θbad) > loss(prob, θ_true)   # responds to viscosity

        # reset_state! makes evaluation repeatable
        @test loss(prob, θ_true) == loss(prob, θ_true)
    end

    @testset "pluggable loss (AbstractLoss)" begin
        sim, se = build_test_sim()
        enc = Test2Encoding()
        θ_true = Float32[21.0, -1f6,-1f6,8f5,0.3, 1f6,1f6,8f5,-0.3,
            -1f6,1f6,8f5,0.2, 1f6,-1f6,8f5,-0.2, se.rho_uppermantle, se.rho_litho]
        pts = [CartesianIndex(i, j) for i in 14:18 for j in 15:17][1:8]

        reconstruct!(sim, θ_true, enc)
        FastIsostasy.reset_state!(sim)
        FastIsostasy.init_problem!(sim)
        integ = FastIsostasy.build_integrator(sim)
        FastIsostasy.advance_with_output!(integ, sim, 5f3)
        data = Float32[sim.now.u[p] + sim.now.ue[p] for p in pts]
        obs = Observation(VerticalUpliftObservable(), pts, Float32[5f3], data; σ = 0.1f0)

        # default constructor ⇒ DefaultLoss(), matching the explicit default
        prob_default = ParameterInversion(sim, enc, [obs])
        @test prob_default.lossmodel isa DefaultLoss

        # a custom AbstractLoss rescales the misfit — no other machinery changes
        struct ScaledLoss <: AbstractLoss
            scale::Float32
        end
        FastIsostasy.misfit(l::ScaledLoss, preds, observations) =
            l.scale * FastIsostasy.misfit(DefaultLoss(), preds, observations)

        prob_scaled = ParameterInversion(sim, enc, [obs]; lossmodel = ScaledLoss(3f0))
        @test loss(prob_scaled, θ_true) ≈ 3f0 * loss(prob_default, θ_true)
    end

    @testset "regularization penalties" begin
        sim, se = build_test_sim()
        reconstruct!(sim, Float32[21.0, 0f0,0f0,8f5,0f0, 0f0,0f0,8f5,0f0,
            0f0,0f0,8f5,0f0, 0f0,0f0,8f5,0f0, 3200f0, 2600f0], Test2Encoding())
        θ = Float32[1, 2, 3]
        @test FastIsostasy.penalty(L2Reg(2f0), sim, θ) ≈ 2f0 * 14f0
        # log10 viscosity ≈ 21 is inside [19, 23] ⇒ zero hinge
        @test FastIsostasy.penalty(DecodedBounds(Log10Viscosity(), 19f0, 23f0, 1f0), sim, θ) ≈ 0 atol=1f-3
        # ρ_uppermantle = 3200 is below [3300, 3400] ⇒ positive hinge
        @test FastIsostasy.penalty(DecodedBounds(UpperMantleDensity(), 3300f0, 3400f0, 1f0), sim, θ) > 0
    end

end
