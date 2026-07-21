# Phase-6 item 1: Enzyme through KernelAbstractions kernels on CUDA.
#
# The CPU counterpart of this test lives in `test_ad_validity.jl`. Here we run the
# *same* inversion setup on both backends and require the GPU gradient to match the
# CPU gradient to round-off, which pins down every device-specific AD rule at once:
# the CUFFT plan rules and the reduction shims (`sumabs2`/`totalsum`/`inner`) in
# `ext/FastIsostasyEnzymeCUDAExt.jl`, plus `match_array` for the host coordinate grids.
#
# Finite differences are included as an independent (backend-agnostic) check so that a
# bug present in *both* AD paths cannot pass silently.
#
# Note: the first GPU gradient pays a large Enzyme+CUDA JIT compilation cost (order
# 10 min). That is compile time, not run time.

using FastIsostasy, CUDA, Enzyme, Test

CUDA.allowscalar(false)

# 19-parameter `Test2Encoding` control vector: log10 viscosity background, four
# Gaussian anomalies (x, y, width, amplitude), then the two densities.
const GPU_AD_THETA = Float64[
    21.0,
    -1.0e6, -1.0e6, 8.0e5,  0.3,
     1.0e6,  1.0e6, 8.0e5, -0.3,
    -1.0e6,  1.0e6, 8.0e5,  0.2,
     1.0e6, -1.0e6, 8.0e5, -0.2,
]

"""
Build a `ParameterInversion` on `arraykernel` whose observations are the model's own
output at `GPU_AD_THETA`, so the truth is reproducible on either backend.
"""
function ad_validity_gpu_setup(arraykernel; n = 4, dt = 100.0, tend = 400.0)
    W = 3.0e6
    domain = RegionalDomain(W, n; arraykernel = arraykernel)

    H0 = zeros(domain)
    H1 = 1.0e3 .* (domain.R .< 1.0e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = false, transition = SmoothTransition(10.0),
        diffeq = DiffEqOptions(alg = EulerIntegrator(), dt_min = dt))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, tend); opts = opts, nout = nout)

    enc = Test2Encoding()
    θ = vcat(GPU_AD_THETA, se.rho_uppermantle, se.rho_litho)
    pts = [CartesianIndex(i, j) for i in 6:9 for j in 6:8][1:8]

    # Generate the synthetic truth by running the forward model at θ.
    obs0 = Observation(VerticalUpliftObservable(), pts, [tend], zeros(length(pts)); σ = 0.1)
    p0 = ParameterInversion(sim, enc, [obs0])
    reconstruct!(sim, θ, enc)
    preds = FastIsostasy.allocate_predictions(p0)
    FastIsostasy.forward_predict!(preds, p0)

    obs = Observation(VerticalUpliftObservable(), pts, [tend], copy(preds[1]); σ = 0.1)
    return ParameterInversion(sim, enc, [obs]), θ
end

# Step size relative to the component magnitude: θ spans O(0.1) Gaussian
# amplitudes to O(1e3) densities, and an absolute ε would be a ~1e-9 relative
# perturbation on the latter — pure cancellation noise.
function central_difference(prob, θ, i; ε_rel = 1.0e-5)
    ε = ε_rel * max(1.0, abs(θ[i]))
    e = zeros(length(θ)); e[i] = 1.0
    return (loss(prob, θ .+ ε .* e) - loss(prob, θ .- ε .* e)) / (2ε)
end

@testset "gpu forward-mode AD validity" begin
    prob_cpu, θ = ad_validity_gpu_setup(Array)
    prob_gpu, _ = ad_validity_gpu_setup(CuArray)

    # Perturb off the truth so the gradient is non-zero.
    θ0 = copy(θ); θ0[1] += 0.15

    # The forward model itself must agree before differentiating it.
    @test loss(prob_gpu, θ0) ≈ loss(prob_cpu, θ0) rtol = 1e-12

    g_cpu = similar(θ0); FastIsostasy.gradient!(g_cpu, prob_cpu, θ0)
    g_gpu = similar(θ0); FastIsostasy.gradient!(g_gpu, prob_gpu, θ0)

    # GPU vs CPU AD: same algorithm, so this is a round-off-level comparison.
    @test g_gpu ≈ g_cpu rtol = 1e-10

    # GPU AD vs finite differences on a spread of components (background viscosity,
    # two Gaussian coordinates, and the two densities) as an independent check.
    for i in (1, 5, 9, 18, 19)
        @test g_gpu[i] ≈ central_difference(prob_gpu, θ0, i) rtol = 1e-4 atol = 1e-7
    end
end
