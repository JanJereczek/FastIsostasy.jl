# Phase-2 AD validity (the go/no-go): forward-mode Enzyme gradient of the full
# inversion `loss` — reconstruct! → fixed-step EulerIntegrator run → data misfit — through
# the explicit lat-variable Maxwell path, validated against central finite
# differences. Also exercises `gradient!(::TangentMode)` end to end.

using FastIsostasy
using Enzyme
using Test

function build_ad_prob(; n = 5, dt = 100.0, tend = 400.0)
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

    # synthetic uplift ground truth at t_end, via the same forward path as `loss`
    obs0 = Observation(VerticalUpliftObservable(), pts, [tend], zeros(length(pts)); σ = 0.1)
    prob0 = ParameterInversion(sim, enc, [obs0])
    reconstruct!(sim, θ_true, enc)
    preds = FastIsostasy.allocate_predictions(prob0)
    FastIsostasy.forward_predict!(preds, prob0)
    data = copy(preds[1])

    obs = Observation(VerticalUpliftObservable(), pts, [tend], data; σ = 0.1)
    return ParameterInversion(sim, enc, [obs]), θ_true
end

# central FD directional derivative of loss along dθ
function fd_dir(prob, θ, dθ; ε = 1e-5)
    return (loss(prob, θ .+ ε .* dθ) - loss(prob, θ .- ε .* dθ)) / (2ε)
end

@testset "AD validity (go/no-go)" begin
    prob, θ_true = build_ad_prob()

    # loss ≈ 0 at the truth (direct-Euler forward reproduces the synthetic data)
    @test loss(prob, θ_true) < 1e-6

    # perturb the viscosity block so the loss and its gradient are nontrivial
    θ0 = copy(θ_true); θ0[1] += 0.15
    @test loss(prob, θ0) > 1.0

    # directional derivative vs FD
    dθ = zeros(length(θ0)); dθ[1] = 1.0; dθ[5] = 0.3; dθ[9] = -0.2
    mode = Enzyme.set_runtime_activity(Enzyme.Forward)
    d_ad = only(Enzyme.autodiff(mode, loss,
        Duplicated(prob, Enzyme.make_zero(prob)), Duplicated(θ0, dθ)))
    @test isapprox(d_ad, fd_dir(prob, θ0, dθ); rtol = 1e-5)

    # full gradient via gradient!(::TangentMode) vs component-wise FD on the
    # viscosity block (indices 1 and 5,9,13,17 = log10η background + Gaussian amps)
    g = similar(θ0)
    FastIsostasy.gradient!(g, prob, θ0)
    for i in (1, 5, 9)
        e = zeros(length(θ0)); e[i] = 1.0
        @test isapprox(g[i], fd_dir(prob, θ0, e); rtol = 1e-5)
    end
end
