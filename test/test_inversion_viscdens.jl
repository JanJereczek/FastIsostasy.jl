# Test 2 (docs example "Inverse calibration"): calibrate the solid-Earth
# parameters — a 4-Gaussian mantle-viscosity field + two densities — from
# full-field vertical-uplift observations under a *known* ice load, via
# `ParameterInversion` + forward-mode AD + L-BFGS. Ground truth = a forward run
# with known θ; the inversion must reproduce the data, match FD gradients,
# decrease the loss, and recover the viscosity field and densities. Lean CI
# counterpart of docs/src/examples/inverse_calibration.jl.

using FastIsostasy
using Enzyme
using Optim
using Random: seed!
using Test

## a radially-symmetric Vialov dome (known, fixed ice load)
function vialov_dome(domain, xc, yc, L, Hc)
    r = @. sqrt((domain.X - xc)^2 + (domain.Y - yc)^2)
    return @. Hc * max(1 - (r / L)^(4 // 3), 0)^(3 // 8)
end

function build_viscdens_problem()
    W, n = 3.0e6, 5
    domain = RegionalDomain(W, n)
    knot_times = [0.0, 5.0e3, 8.0e3, 9.0e3, 1.0e4]
    t_span = (0.0, 1.0e4)

    gscale = [1.0e6, 1.0e6, 1.0e6, 1.0]
    scales = vcat(1.0, repeat(gscale, 4), 1.0e3, 1.0e3)
    enc = Test2Encoding(scale = scales)
    θ_phys = [21.0,
        -1.0e6, -1.0e6, 8.0e5,  0.5,
         1.0e6,  1.0e6, 8.0e5, -0.5,
        -1.0e6,  1.0e6, 7.0e5,  0.3,
         1.0e6, -1.0e6, 7.0e5, -0.3,
        3400.0, 3200.0]
    θ_true = θ_phys ./ scales

    H_dome = vialov_dome(domain, 0.0, 0.0, 2.0e6, 2.5e3)
    H_snapshots = [s .* H_dome for s in [0.0, 0.4, 1.0, 0.5, 0.0]]
    it = TimeInterpolatedIceThickness(knot_times, H_snapshots, domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = false, transition = SmoothTransition(10.0),
        diffeq = DiffEqOptions(alg = EulerIntegrator(), dt_min = 500.0))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, t_span; opts = opts, nout = nout)

    pts = [CartesianIndex(i, j) for i in 4:29 for j in 4:29]
    obs_times = [2.0e3, 4.0e3, 6.0e3, 8.0e3, 1.0e4]
    obs0 = Observation(VerticalUpliftObservable(), pts, obs_times,
        zeros(length(pts) * length(obs_times)); σ = 0.5)
    prob_gt = ParameterInversion(sim, enc, [obs0])
    reconstruct!(sim, θ_true, enc)
    preds = FastIsostasy.allocate_predictions(prob_gt)
    FastIsostasy.forward_predict!(preds, prob_gt)
    data = copy(preds[1])
    obs = Observation(VerticalUpliftObservable(), pts, obs_times, data; σ = 0.5)
    return ParameterInversion(sim, enc, [obs]), θ_true, enc, scales, sim
end

fd_dir(p, θ, e; ε = 1e-4) = (loss(p, θ .+ ε .* e) - loss(p, θ .- ε .* e)) / (2ε)

@testset "inverse calibration (viscosity + densities)" begin
    prob, θ_true, enc, scales, sim = build_viscdens_problem()

    # loss vanishes at the truth
    @test loss(prob, θ_true) < 1e-6

    # informed initial guess (dimensionless, ~10% off)
    seed!(3)
    θ0 = copy(θ_true)
    θ0[1]      += 0.15 * randn()
    θ0[2:17]  .+= 0.10 .* randn(16)
    θ0[18:19] .+= 0.10 .* randn(2)
    @test loss(prob, θ0) > 1.0

    # AD gradient matches central FD (background viscosity, a Gaussian centre,
    # a density)
    g = similar(θ0)
    FastIsostasy.gradient!(g, prob, θ0)
    for i in (1, 2, 18)
        e = zeros(19); e[i] = 1.0
        @test isapprox(g[i], fd_dir(prob, θ0, e); rtol = 1e-4)
    end

    # invert and check the loss collapsed
    result = solve!(prob, θ0; optimizer = LBFGS(), iterations = 60)
    θ_hat = Optim.minimizer(result)
    @test loss(prob, θ_hat) < loss(prob, θ0)
    @test loss(prob, θ_hat) < 1e-3 * loss(prob, θ0)

    # viscosity field recovery
    reconstruct!(sim, θ_true, enc)
    logη_true = log10.(copy(sim.solidearth.effective_viscosity))
    reconstruct!(sim, θ_hat, enc)
    logη_hat = log10.(copy(sim.solidearth.effective_viscosity))
    @test maximum(abs.(logη_hat .- logη_true)) < 0.05      # decades

    # density recovery (well-identified by the rich full-field data)
    php = θ_hat .* scales; ptp = θ_true .* scales
    @test abs(php[18] - ptp[18]) < 50.0                    # ρ_uppermantle, kg/m³
    @test abs(php[19] - ptp[19]) < 50.0                    # ρ_litho, kg/m³
end
