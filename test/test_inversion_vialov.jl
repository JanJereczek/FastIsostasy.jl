# Test 1 (docs example "Inverse ice history"): joint ice-history + bimodal
# viscosity inversion via `IceLoadInversion` + forward-mode AD + L-BFGS. Ground
# truth = a forward run with known θ; the inversion must reproduce the data,
# match FD gradients, decrease the loss, and recover the ice/viscosity fields.
# This is the lean CI counterpart of docs/src/examples/inverse_ice_history.jl.

using FastIsostasy
using Enzyme
using Optim
using Random: randperm, seed!
using Test

function build_vialov_problem()
    W, n = 3.0e6, 5
    domain = RegionalDomain(W, n)
    knot_times = [0.0, 5.0e3, 8.0e3, 9.0e3, 1.0e4]
    K = length(knot_times)
    t_span = (0.0, 1.0e4)
    radii = (0.8e6, 0.7e6, 0.7e6)
    visc_amps = (-1.0, 1.0)
    scales = vcat(fill(2.0e3, 3K), fill(1.0e6, 6), 1.0, fill(1.0e6, 6))
    enc = Test1Encoding(knot_times, radii, visc_amps; scale = scales)

    sawtooth(peak) = [0.0, 0.4, 1.0, 0.5, 0.0] .* peak
    Hc_phys = vcat(sawtooth(2.5e3), sawtooth(2.0e3), sawtooth(1.8e3))
    centers_phys = [-1.3e6, -1.3e6, 1.3e6, 1.3e6, 1.3e6, -1.3e6]
    visc_phys = [21.0, -1.0e6, -1.0e6, 8.0e5, 1.0e6, 1.0e6, 8.0e5]
    θ_true = vcat(Hc_phys, centers_phys, visc_phys) ./ scales

    it = TimeInterpolatedIceThickness(knot_times,
        [zeros(domain) for _ in knot_times], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = false, transition = SmoothTransition(10.0),
        diffeq = DiffEqOptions(alg = FIEuler(), dt_min = 500.0))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, t_span; opts = opts, nout = nout)

    seed!(1234)
    allidx = vec([CartesianIndex(i, j) for i in 4:29, j in 4:29])
    pts = allidx[randperm(length(allidx))[1:24]]
    obs_times = collect(1.0e3:1.0e3:1.0e4)

    obs0 = Observation(VerticalUpliftObservable(), pts, obs_times,
        zeros(length(pts) * length(obs_times)); σ = 0.5)
    prob_gt = IceLoadInversion(sim, enc, [obs0])
    reconstruct!(sim, θ_true, enc)
    preds = FastIsostasy.allocate_predictions(prob_gt)
    FastIsostasy.forward_predict!(preds, prob_gt)
    data = copy(preds[1])
    obs = Observation(VerticalUpliftObservable(), pts, obs_times, data; σ = 0.5)
    prob = IceLoadInversion(sim, enc, [obs])
    return prob, θ_true, enc, scales, K, sim
end

fd_dir(p, θ, e; ε = 1e-4) = (loss(p, θ .+ ε .* e) - loss(p, θ .- ε .* e)) / (2ε)

@testset "inverse ice history (Vialov)" begin
    prob, θ_true, enc, scales, K, sim = build_vialov_problem()

    # loss vanishes at the truth (AD forward reproduces the synthetic data)
    @test loss(prob, θ_true) < 1e-6

    # informed initial guess (dimensionless, ~10% off)
    seed!(2)
    θ0 = copy(θ_true)
    θ0[1:3K]      .+= 0.12 .* randn(3K)
    θ0[3K+1:3K+6] .+= 0.08 .* randn(6)
    θ0[3K+7]       += 0.15 * randn()
    θ0[3K+8:end]  .+= 0.08 .* randn(6)
    @test loss(prob, θ0) > 1.0

    # AD gradient matches central FD on a few components (ice knot, dome centre,
    # background viscosity)
    g = similar(θ0)
    FastIsostasy.gradient!(g, prob, θ0)
    for i in (1, 3K + 1, 3K + 7)
        e = zeros(length(θ0)); e[i] = 1.0
        @test isapprox(g[i], fd_dir(prob, θ0, e); rtol = 1e-4)
    end

    # invert and check the loss dropped by orders of magnitude
    result = solve!(prob, θ0; optimizer = LBFGS(), iterations = 80)
    θ_hat = Optim.minimizer(result)
    @test loss(prob, θ_hat) < loss(prob, θ0)
    @test loss(prob, θ_hat) < 1e-1 * loss(prob, θ0)

    # field-level recovery (the physically meaningful check)
    reconstruct!(sim, θ_true, enc)
    H_true = copy(FastIsostasy.ice_snapshots(sim)[3])
    logη_true = log10.(copy(sim.solidearth.effective_viscosity))
    reconstruct!(sim, θ_hat, enc)
    H_hat = copy(FastIsostasy.ice_snapshots(sim)[3])
    logη_hat = log10.(copy(sim.solidearth.effective_viscosity))
    @test maximum(abs.(H_hat .- H_true)) < 200.0            # peak ice ≈ 2500 m
    @test maximum(abs.(logη_hat .- logη_true)) < 0.1        # decades
end
