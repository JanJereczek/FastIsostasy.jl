# Tests for `SimulatedObservable` (roadmap §4c item 2): forward-run virtual
# stations attached to a plain `Simulation`, recorded by `run!` without writing
# full 2-D fields to output.

using FastIsostasy
using Test

function build_simobs_test_sim(; n = 5, tend = 5f3)
    W = 3f6
    domain = RegionalDomain(W, n)
    H0 = zeros(domain)
    H1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0f0, 1f0, 50f3], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88f3], layer_viscosities = [1f21])
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0, tend))
    return sim
end

@testset "SimulatedObservable" begin

    @testset "attach_simobs! records the same values as a manual extraction" begin
        sim = build_simobs_test_sim()
        pts = [CartesianIndex(3, 3), CartesianIndex(4, 4)]
        times = Float32[1f3, 5f3]

        so = attach_simobs!(sim, VerticalUpliftObservable(), pts, times)
        @test length(sim.simobs) == 1
        @test so.k == 1

        run!(sim)

        # Cursor exhausted, data filled at every (point, time) pair.
        @test so.k == length(times) + 1
        @test length(so.data) == length(pts) * length(times)

        # Cross-check against an independent full-field extraction at the same
        # final time (run! leaves sim.now at t_span[2] == times[end]).
        expected = Float32[sim.now.u[p] + sim.now.ue[p] for p in pts]
        @test so.data[(end - length(pts) + 1):end] ≈ expected
    end

    @testset "no simobs attached ⇒ run! unaffected" begin
        sim = build_simobs_test_sim()
        @test isempty(sim.simobs)
        run!(sim)   # should not error, and needs no simobs bookkeeping
        @test isempty(sim.simobs)
    end

    @testset "multiple stations + tags, independent cursors" begin
        sim = build_simobs_test_sim()
        pts1 = [CartesianIndex(3, 3)]
        pts2 = [CartesianIndex(4, 4), CartesianIndex(2, 2)]
        so1 = attach_simobs!(sim, VerticalUpliftObservable(), pts1, Float32[2f3])
        so2 = attach_simobs!(sim, VerticalUpliftRateObservable(), pts2, Float32[1f3, 3f3, 5f3])

        run!(sim)

        @test so1.k == 2 && length(so1.data) == 1
        @test so2.k == 4 && length(so2.data) == 6
    end

end
