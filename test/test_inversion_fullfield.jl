# Test 3 (full-field): invert the *entire* 2-D effective-viscosity field with no
# encoding (`encoding === nothing`; θ is the flattened log10-viscosity field) via
# the checkpointed reverse-mode adjoint (`AdjointMode`), with a Tikhonov
# regularization that flows *through the sim* (the reg-gradient-through-sim path
# deferred to Test 3). Validates: the adjoint gradient matches central FD and
# forward-mode Enzyme (TangentMode) component-for-component; `loss_and_gradient!`
# returns the primal; and a short L-BFGS run moves the field toward the truth up
# to the regularization bias.
#
# Regularizer choice — `TikhonovReg(FieldTarget(Log10Viscosity()), Order0())`:
# because θ *is* the log10-viscosity field here, this magnitude penalty
# λ·Σ(log10η)² is numerically the same object as the roadmap's `L2Reg` (=
# `TikhonovReg(ThetaTarget(), Order0())`), so it honors the roadmap's intent — but
# routing it through `FieldTarget`/`decoded(Log10Viscosity, sim)` means the penalty
# reads the *sim's* `effective_viscosity`, exercising exactly the reg-gradient-
# through-sim reverse Test 3 is meant to validate (plain `ThetaTarget` L2Reg never
# touches the sim). `Order1` smoothness was rejected: on this coarse (~750 km) grid
# the FD stencils divide by the large physical `dx`, so ‖∇log10η‖² is ~1e-15 —
# below the cross-check tolerances, hence untested. Core `L2Reg`/`ThetaTarget` are
# unchanged; this is a test-side choice.

using FastIsostasy
using Enzyme
using Checkpointing
using Optim
using Test

# self-contained log10 Gaussian bump (avoids depending on the un-exported builder)
_bump(X, Y, μx, μy, σ, amp) = @. amp * exp(-((X - μx)^2 + (Y - μy)^2) / (2 * σ^2))

function build_fullfield_problem(; n = 3, W = 3.0e6, dt = 100.0, tend = 400.0,
        obstimes = [200.0, 400.0])
    domain = RegionalDomain(W, n)

    # known, fixed ice load (central cap) — same shape as test_adjoint_validity
    H0 = zeros(domain)
    H1 = 1.0e3 .* (domain.R .< 1.5e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = false, transition = SmoothTransition(10.0),
        diffeq = DiffEqOptions(alg = EulerIntegrator(), dt_min = dt))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, tend);
        opts = opts, nout = nout)

    # ground-truth log10-viscosity field: background + 4 Gaussian anomalies
    X, Y = domain.X, domain.Y
    logη = fill(21.0, size(X))
    logη .+= _bump(X, Y, -1.0e6, -1.0e6, 8.0e5,  0.3)
    logη .+= _bump(X, Y,  1.0e6,  1.0e6, 8.0e5, -0.3)
    logη .+= _bump(X, Y, -1.0e6,  1.0e6, 8.0e5,  0.2)
    logη .+= _bump(X, Y,  1.0e6, -1.0e6, 8.0e5, -0.2)
    θ_true = vec(copy(logη))          # full-field control, flattened column-major

    # dense-ish interior observation of vertical uplift at two times
    pts = [CartesianIndex(i, j) for i in 2:7 for j in 2:7]
    reg = TikhonovReg(FieldTarget(Log10Viscosity()), Order0(); λ = 0.01)

    nt = length(obstimes)
    obs0 = Observation(VerticalUpliftObservable(), pts, obstimes,
        zeros(length(pts) * nt); σ = 0.05)
    prob0 = ParameterInversion(sim, nothing, [obs0];
        regularizations = (reg,), diffmode = AdjointMode())
    reconstruct!(sim, θ_true, nothing)
    preds = FastIsostasy.allocate_predictions(prob0)
    FastIsostasy.forward_predict!(preds, prob0)
    data = copy(preds[1])

    obs = Observation(VerticalUpliftObservable(), pts, obstimes, data; σ = 0.05)
    prob = ParameterInversion(sim, nothing, [obs];
        regularizations = (reg,), diffmode = AdjointMode())
    return prob, θ_true
end

fd_dir(prob, θ, dθ; ε = 1.0e-5) =
    (loss(prob, θ .+ ε .* dθ) - loss(prob, θ .- ε .* dθ)) / (2ε)

@testset "full-field viscosity inversion (AdjointMode)" begin
    prob, θ_true = build_fullfield_problem()
    nθ = length(θ_true)
    @test nθ == 64                        # 8×8 grid, one param per cell

    # data reproduced at the truth: the *data misfit* vanishes (the loss itself
    # floors at the nonzero Tikhonov penalty on the true field, so check the misfit
    # directly via the forward machinery rather than the full loss)
    reconstruct!(prob.sim, θ_true, nothing)
    preds = FastIsostasy.allocate_predictions(prob)
    FastIsostasy.forward_predict!(preds, prob)
    @test FastIsostasy.misfit(prob.lossmodel, preds, prob.observations) < 1.0e-6

    # perturbed start: bump a handful of interior cells' log-viscosity
    θ0 = copy(θ_true)
    for i in (20, 28, 29, 36, 37, 44)
        θ0[i] += 0.3
    end
    @test loss(prob, θ0) > 1.0

    # adjoint gradient of the full loss (data misfit + reg-through-sim)
    g_adj = similar(θ0)
    FastIsostasy.gradient!(g_adj, prob, θ0, AdjointMode())

    # component-wise vs central FD on cells spanning perturbed/unperturbed,
    # ice-covered/edge (rtol on the bulk, atol to tolerate near-zero components)
    for i in (20, 28, 29, 36, 37, 44, 1, 64)
        e = zeros(nθ); e[i] = 1.0
        @test isapprox(g_adj[i], fd_dir(prob, θ0, e); rtol = 1.0e-4, atol = 1.0e-7)
    end

    # full-vector agreement with forward-mode Enzyme (TangentMode = nθ passes) —
    # cross-checks every component, including the reg-through-sim contribution
    g_fwd = similar(θ0)
    FastIsostasy.gradient!(g_fwd, prob, θ0, TangentMode())
    @test isapprox(g_adj, g_fwd; rtol = 1.0e-5)

    # loss_and_gradient! returns the primal alongside the same gradient
    g2 = similar(θ0)
    l2 = FastIsostasy.loss_and_gradient!(g2, prob, θ0, AdjointMode())
    @test l2 ≈ loss(prob, θ0)
    @test g2 == g_adj

    # short L-BFGS run: loss decreases and the field moves toward the truth
    # (recovery up to the Tikhonov shrinkage bias)
    l0 = loss(prob, θ0)
    err0 = sum(abs2, θ0 .- θ_true)
    res = solve!(prob, θ0; optimizer = LBFGS(), iterations = 12)
    θhat = Optim.minimizer(res)
    @test loss(prob, θhat) < l0
    @test sum(abs2, θhat .- θ_true) < 0.8 * err0
end
