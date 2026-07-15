# Static-analysis gates via JET.
#
# 1. Whole-package error analysis (`report_package`): undefined variables,
#    guaranteed MethodErrors, may-be-undefined locals, etc. Two report classes
#    are expected and filtered out (predicates below):
#    (a) GPU-branch KernelAbstractions launches — `Kernel{<:GPU}` has no call
#        method unless a GPU backend package (e.g. CUDA) is loaded, and those
#        branches are only reachable with device arrays, in which case the
#        backend is loaded. (Dual-path design, see src/derivatives.jl.)
#    (b) `load_dataset` forwards `kwargs...` to loaders that accept none;
#        passing kwargs for those datasets is supposed to throw.
#    Anything else is a regression: fix it rather than widening the filters.
#
# 2. `report_opt` on the RHS hot path (`update_diagnostics!`, explicit
#    laterally-variable Maxwell configuration): asserts zero runtime dispatch
#    and no captured-variable boxes. This is the path Enzyme differentiates;
#    dynamic dispatch there is both a perf bug and an AD hazard
#    (see roadmaps/ad_inversion.md). Baseline on adoption (2026-07-15): clean.

using JET

_report_sig(r) = r isa JET.MethodErrorReport ? string(r.t) : ""
_is_gpu_kernel_launch(r) = occursin("KernelAbstractions.Kernel", _report_sig(r))
_is_loader_kwcall(r) = occursin("kwcall", _report_sig(r)) && occursin("load_", _report_sig(r))
_expected_noise(r) = _is_gpu_kernel_launch(r) || _is_loader_kwcall(r)

@testset "JET package error analysis" begin
    rep = JET.report_package(FastIsostasy; target_modules = (FastIsostasy,))
    unexpected = filter(!_expected_noise, JET.get_reports(rep))
    if !isempty(unexpected)
        show(stdout, MIME"text/plain"(), rep)  # full report for debugging
    end
    @test isempty(unexpected)
end

@testset "JET type stability of the RHS hot path" begin
    W = 3.0e6
    domain = RegionalDomain(W, 4)
    H0 = zeros(domain)
    H1 = 1.0e3 .* (domain.R .< 1.0e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = false, transition = SmoothTransition(10.0),
        diffeq = DiffEqOptions(alg = FIEuler(), dt_min = 100.0))
    nout = FastIsostasy.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    sim = Simulation(domain, bcs, RegionalSeaLevel(), se, (0.0, 400.0);
        opts = opts, nout = nout)

    dudt = copy(sim.now.dudt)
    u = copy(sim.now.u)
    opt = JET.report_opt(update_diagnostics!, Base.typesof(dudt, u, sim, 0.0);
        target_modules = (FastIsostasy,))
    if !isempty(JET.get_reports(opt))
        show(stdout, MIME"text/plain"(), opt)
    end
    @test isempty(JET.get_reports(opt))
end
