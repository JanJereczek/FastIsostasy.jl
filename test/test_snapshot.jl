# State-snapshot machinery (roadmap Phase 5): `snapshot!`/`restore!` must capture
# *all* mutated state (arrays + scalars like `count_sparse_updates`, `z_bsl`, the
# BSL and clock) so that restoring and re-running a forward interval reproduces
# the trajectory bit-for-bit — the foundation of the checkpointed adjoint.

using FastIsostasy
using Test

function build_snapshot_sim(; n = 5, tend = 5.0e3)
    W = 3.0e6
    domain = RegionalDomain(W, n)
    H0 = zeros(domain); H1 = 1.0e3 .* (domain.R .< 1.0e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; show_progress = false,
        integ = EulerIntegrator(dt = 250.0))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    return Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, tend);
        opts = opts, nout = nout)
end

@testset "state snapshot / restore" begin
    sim = build_snapshot_sim()

    # snapshot the initial state, then advance
    buf = StateSnapshot(sim)
    snapshot!(buf, sim)
    run!(sim)

    u1 = copy(sim.now.u)
    ue1 = copy(sim.now.ue)
    zss1 = copy(sim.now.z_ss)
    csu1 = sim.now.count_sparse_updates
    bslz1 = sim.sealevel.bsl.z
    @test sim.timer.t == 5.0e3          # advanced

    # restore rewinds every captured piece of state
    restore!(sim, buf)
    @test sim.timer.t == 0.0
    @test sim.now.count_sparse_updates == 0
    @test sim.now.u == sim.ref.u        # back to the initial condition

    # re-running from the restored state reproduces the trajectory bit-for-bit
    run!(sim)
    @test sim.now.u == u1
    @test sim.now.ue == ue1
    @test sim.now.z_ss == zss1
    @test sim.now.count_sparse_updates == csu1
    @test sim.sealevel.bsl.z == bslz1

    # snapshotting mid-run and restoring it also round-trips exactly
    sim2 = build_snapshot_sim()
    run!(sim2)                          # bring it to a nontrivial state
    mid = StateSnapshot(sim2)
    snapshot!(mid, sim2)
    umid = copy(sim2.now.u)
    # perturb the live state, then restore
    sim2.now.u .+= 1.0
    sim2.now.count_sparse_updates += 5
    restore!(sim2, mid)
    @test sim2.now.u == umid
    @test sim2.now.count_sparse_updates == mid.now.count_sparse_updates
end
