# Phase-5 forward recording (roadmap §6): `record_forward!` must reproduce
# `forward_predict!` bit for bit while logging the accepted (t, dt) sequence per
# save interval and snapshotting each interval boundary; `replay_interval!` from
# a checkpoint with the frozen dt-sequence must land exactly on the next
# checkpoint's state.

using FastIsostasy
using Test

function build_recording_prob(; n = 5, dt = 100.0, tend = 400.0,
        alg = EulerIntegrator(dt = dt), obstimes = [100.0, 250.0, 400.0])
    W = 3.0e6
    domain = RegionalDomain(W, n)
    H0 = zeros(domain); H1 = 1.0e3 .* (domain.R .< 1.0e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; show_progress = false, integ = alg)
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, tend);
        opts = opts, nout = nout)

    pts = [CartesianIndex(i, j) for i in 14:18 for j in 15:17][1:8]
    data = zeros(length(pts) * length(obstimes))
    obs = Observation(VerticalUpliftObservable(), pts, obstimes, data; σ = 0.1)
    # no encoding needed to exercise the forward recording; AdjointMode (its
    # consumer) is the mode that allows encoding === nothing
    return ParameterInversion(sim, nothing, [obs]; diffmode = AdjointMode())
end

# Bitwise comparison of two StateSnapshots (arrays with ==, scalars with ===,
# BSL Real fields recursively) — the replay acceptance criterion.
function bsl_matches(a, b)
    for f in fieldnames(typeof(a))
        va, vb = getfield(a, f), getfield(b, f)
        if va isa Real
            va === vb || return false
        elseif va isa FastIsostasy.AbstractBSL
            bsl_matches(va, vb) || return false
        end
    end
    return true
end

function snapshots_match(a::StateSnapshot, b::StateSnapshot)
    for f in fieldnames(typeof(a.now))
        va, vb = getfield(a.now, f), getfield(b.now, f)
        if va isa FastIsostasy.ColumnAnomalies || va isa KinematicBSL
            for cf in fieldnames(typeof(va))
                getfield(va, cf) == getfield(vb, cf) || return false
            end
        elseif va isa AbstractArray
            va == vb || return false
        else
            va === vb || return false
        end
    end
    return a.timer_t === b.timer_t && bsl_matches(a.bsl, b.bsl)
end

@testset "forward recording (adjoint Phase 5)" begin

    prob = build_recording_prob()
    sim = prob.sim
    bounds = [0.0; prob.extract_times]

    preds0 = FastIsostasy.allocate_predictions(prob)
    FastIsostasy.forward_predict!(preds0, prob)

    rec = ForwardRecord(prob)
    preds1 = FastIsostasy.allocate_predictions(prob)
    record_forward!(preds1, rec, prob)

    @testset "recording reproduces forward_predict! bitwise" begin
        @test all(preds1[k] == preds0[k] for k in eachindex(preds0))
    end

    @testset "recorded (t, dt) structure" begin
        @test FastIsostasy.nintervals(rec) == 3
        @test length(rec.checkpoints) == 4
        for i in 1:3
            ts, hs = first.(rec.steps[i]), last.(rec.steps[i])
            @test ts[1] == bounds[i]                   # interval starts at boundary
            @test issorted(ts) && all(>(0), hs)
            @test sum(hs) ≈ bounds[i + 1] - bounds[i]  # steps tile the interval
            @test all(ts .+ hs .≈ [ts[2:end]; bounds[i + 1]])
            @test all(hs .<= 100.0)                    # never exceeds dt_min
        end
        # intervals 2 and 3 span 150 with dt = 100 ⇒ final step clipped to 50
        @test last(rec.steps[2])[2] == 50.0
        @test last(rec.steps[3])[2] == 50.0
        # boundary checkpoints carry the boundary times
        @test [cp.timer_t for cp in rec.checkpoints] == bounds
    end

    @testset "replay of a middle interval is bit-identical" begin
        u, dudt = similar(sim.now.u), similar(sim.now.u)
        replay_interval!(u, dudt, rec, prob, 2)
        got = snapshot!(StateSnapshot(sim), sim)
        @test snapshots_match(got, rec.checkpoints[3])
        @test u == rec.checkpoints[3].now.u
    end

    @testset "replay of every interval lands on the next checkpoint" begin
        u, dudt = similar(sim.now.u), similar(sim.now.u)
        for i in 1:3
            replay_interval!(u, dudt, rec, prob, i)
            got = snapshot!(StateSnapshot(sim), sim)
            @test snapshots_match(got, rec.checkpoints[i + 1])
        end
    end

    @testset "re-recording reuses buffers and is deterministic" begin
        steps_before = deepcopy(rec.steps)
        preds2 = FastIsostasy.allocate_predictions(prob)
        record_forward!(preds2, rec, prob)
        @test rec.steps == steps_before
        @test all(preds2[k] == preds1[k] for k in eachindex(preds1))
    end

    @testset "adaptive algorithm: steps recorded, replay deferred" begin
        proba = build_recording_prob(alg = BS3Integrator{Float64}(dt_min = 100.0))
        preds0a = FastIsostasy.allocate_predictions(proba)
        FastIsostasy.forward_predict!(preds0a, proba)

        reca = ForwardRecord(proba)
        predsa = FastIsostasy.allocate_predictions(proba)
        record_forward!(predsa, reca, proba)

        @test all(predsa[k] == preds0a[k] for k in eachindex(preds0a))
        @test all(!isempty, reca.steps)
        for i in 1:3
            hs = last.(reca.steps[i])
            @test all(>(0), hs)
            @test sum(hs) ≈ bounds[i + 1] - bounds[i]
        end
        u, dudt = similar(proba.sim.now.u), similar(proba.sim.now.u)
        @test_throws ErrorException replay_interval!(u, dudt, reca, proba, 1)
    end

    @testset "RKCIntegrator: widened (t, dt, s) entries record and replay bit-identically" begin
        # RKCIntegrator's steplog_entry is a 3-tuple (t, dt, s), unlike every other
        # AbstractIntegrator's (t, dt) — ForwardRecord sizes its step buffers per
        # algorithm via steplog_entry_type. And, being non-FSAL, RKCIntegrator *replays*
        # (roadmap §7): a frozen (t, dt, s) step is a pure function of the state,
        # so replay from a checkpoint must land on the next checkpoint bit for bit.
        probr = build_recording_prob(alg = RKCIntegrator{Float64}(dt_min = 100.0))
        preds0r = FastIsostasy.allocate_predictions(probr)
        FastIsostasy.forward_predict!(preds0r, probr)

        recr = ForwardRecord(probr)
        predsr = FastIsostasy.allocate_predictions(probr)
        record_forward!(predsr, recr, probr)

        @test all(predsr[k] == preds0r[k] for k in eachindex(preds0r))
        @test all(!isempty, recr.steps)
        for i in 1:3
            ts, hs, ss = first.(recr.steps[i]), getindex.(recr.steps[i], 2), last.(recr.steps[i])
            @test all(>(0), hs)
            @test all(>=(2), ss)                        # stage count, always >= 2
            @test sum(hs) ≈ bounds[i + 1] - bounds[i]
        end

        # Frozen replay of every interval lands on the next checkpoint, bitwise.
        ur, dudtr = similar(probr.sim.now.u), similar(probr.sim.now.u)
        for i in 1:3
            replay_interval!(ur, dudtr, recr, probr, i)
            gotr = snapshot!(StateSnapshot(probr.sim), probr.sim)
            @test snapshots_match(gotr, recr.checkpoints[i + 1])
            @test ur == recr.checkpoints[i + 1].now.u
        end
    end

    @testset "mismatched record/problem throws" begin
        prob2 = build_recording_prob(obstimes = [200.0, 400.0])
        preds = FastIsostasy.allocate_predictions(prob2)
        @test_throws DimensionMismatch record_forward!(preds, rec, prob2)
    end

end
