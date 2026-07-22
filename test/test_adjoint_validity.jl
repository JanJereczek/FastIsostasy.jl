# Phase-5 adjoint validity: the checkpointed reverse-mode gradient
# (`gradient!(::AdjointMode)`, FastIsostasyCheckpointingExt) must match both
# central finite differences and the forward-mode (`TangentMode`) gradient on the
# same setup as `test_ad_validity.jl` — reconstruct! → fixed-step EulerIntegrator run →
# data misfit, lat-variable Maxwell, SmoothTransition. Also checks
# `loss_and_gradient!(::AdjointMode)` returns the primal loss alongside g.

using FastIsostasy
using Enzyme
using Checkpointing
using Test

function build_adj_prob(; n = 5, dt = 100.0, tend = 400.0, obstimes = [200.0, 400.0])
    W = 3.0e6
    domain = RegionalDomain(W, n)
    H0 = zeros(domain); H1 = 1.0e3 .* (domain.R .< 1.0e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = false, transition = SmoothTransition(10.0),
        integ = EulerIntegrator(dt = dt))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, tend);
        opts = opts, nout = nout)

    enc = Test2Encoding()
    θ_true = Float64[21.0, -1e6,-1e6,8e5,0.3, 1e6,1e6,8e5,-0.3,
        -1e6,1e6,8e5,0.2, 1e6,-1e6,8e5,-0.2, se.rho_uppermantle, se.rho_litho]
    pts = [CartesianIndex(i, j) for i in 14:18 for j in 15:17][1:8]

    # synthetic uplift ground truth at multiple times (exercises multi-interval
    # checkpointing: the reverse sweep threads across >1 observation interval)
    nt = length(obstimes)
    obs0 = Observation(VerticalUpliftObservable(), pts, obstimes,
        zeros(length(pts) * nt); σ = 0.1)
    prob0 = ParameterInversion(sim, enc, [obs0])
    reconstruct!(sim, θ_true, enc)
    preds = FastIsostasy.allocate_predictions(prob0)
    FastIsostasy.forward_predict!(preds, prob0)
    data = copy(preds[1])

    obs = Observation(VerticalUpliftObservable(), pts, obstimes, data; σ = 0.1)
    return ParameterInversion(sim, enc, [obs]), θ_true
end

fd_dir(prob, θ, dθ; ε = 1e-5) =
    (loss(prob, θ .+ ε .* dθ) - loss(prob, θ .- ε .* dθ)) / (2ε)

@testset "adjoint validity (AdjointMode)" begin
    prob, θ_true = build_adj_prob()
    θ0 = copy(θ_true); θ0[1] += 0.15
    @test loss(prob, θ0) > 1.0

    g_adj = similar(θ0)
    FastIsostasy.gradient!(g_adj, prob, θ0, AdjointMode())

    # component-wise vs FD on the viscosity block + both densities
    for i in (1, 5, 9, 18, 19)
        e = zeros(length(θ0)); e[i] = 1.0
        @test isapprox(g_adj[i], fd_dir(prob, θ0, e); rtol = 1e-4)
    end

    # full-vector agreement with forward-mode Enzyme (TangentMode)
    g_fwd = similar(θ0)
    FastIsostasy.gradient!(g_fwd, prob, θ0, TangentMode())
    @test isapprox(g_adj, g_fwd; rtol = 1e-5)

    # loss_and_gradient! returns the primal and the same gradient
    g2 = similar(θ0)
    l2 = FastIsostasy.loss_and_gradient!(g2, prob, θ0, AdjointMode())
    @test l2 ≈ loss(prob, θ0)
    @test g2 == g_adj
end
