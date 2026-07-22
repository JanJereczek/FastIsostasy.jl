# Forward-run progress reporting (src/progress.jl): the bar must be driven by
# the stepper, be throttled by *wall* time (its diagnostics reduce over the whole
# grid, so a per-step refresh would be a real cost), and never disturb a run.

using FastIsostasy
using Test

const FI = FastIsostasy

function build_progress_sim(; tspan = (0.0, 400.0), verbose = true, dt_walltime = 0.5)
    W, n = 3.0e6, 5
    domain = RegionalDomain(W, n)
    H0 = zeros(domain)
    H1 = 1.0e3 .* (domain.R .< 1.0e6)
    it = TimeInterpolatedIceThickness([0.0, 1.0, 5.0e4], [H0, H1, H1], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    se = SolidEarth(domain; lithosphere = LaterallyVariableLithosphere(),
        layer_boundaries = [88.0e3], layer_viscosities = [1.0e21])
    opts = SolverOptions(; verbose = verbose, dt_walltime = dt_walltime,
        integ = EulerIntegrator(dt = 50.0))
    nout = FI.NativeOutput(t = Float64[], vars = Symbol[], T = Float64)
    return Simulation(domain, bcs, RegionalSeaLevel(), se, tspan;
        opts = opts, nout = nout)
end

@testset "forward progress reporting" begin

    @testset "tick maps the time span onto the bar" begin
        sim = build_progress_sim(tspan = (0.0, 400.0))
        p = FI.ForwardProgress(sim)
        @test FI.progress_tick(p, 0.0) == 0
        @test FI.progress_tick(p, 200.0) == FI.PROGRESS_TICKS ÷ 2
        # Capped one below the end: only `finish_progress!` completes the bar, so
        # ProgressMeter's completion line is printed exactly once.
        @test FI.progress_tick(p, 400.0) == FI.PROGRESS_TICKS - 1
        @test FI.progress_tick(p, 1e6) == FI.PROGRESS_TICKS - 1
        @test FI.progress_tick(p, -1e6) == 0
    end

    @testset "a zero-length time span does not divide by zero" begin
        sim = build_progress_sim(tspan = (7.0, 7.0))
        p = FI.ForwardProgress(sim)
        @test isfinite(p.t_span) && p.t_span > 0
        @test FI.progress_tick(p, 7.0) isa Int
    end

    @testset "reported values cover the requested diagnostics" begin
        sim = build_progress_sim()
        FI.init_problem!(sim)
        integ = FI.build_integrator(sim)
        vals = FI.progress_values(sim, integ)
        labels = first.(vals)
        @test any(l -> occursin("sim time", l), labels)
        @test any(l -> occursin("time step", l), labels)
        @test any(l -> occursin("viscous displacement u", l), labels)
        @test any(l -> occursin("elastic displacement ue", l), labels)
        @test any(l -> occursin("dz_ss", l), labels)
        @test any(l -> occursin("bsl", l), labels)
        @test length(vals) == 6
        @test all(v -> !isempty(string(last(v))), vals)
    end

    @testset "refresh is throttled by wall time" begin
        sim = build_progress_sim(dt_walltime = 1000.0)   # far longer than the test
        FI.init_problem!(sim)
        integ = FI.build_integrator(sim)

        # ProgressMeter binds its output stream when the bar is built, so the
        # redirect has to cover the construction, not just the drawing.
        p = redirect_stderr(devnull) do
            q = FI.ForwardProgress(sim)
            # Constructed backdated, so the first call always draws (a run
            # shorter than dt_walltime must still show a bar).
            FI.report_progress!(q, integ)
            q
        end
        drawn_at = p.t_wall_last
        @test drawn_at > 0

        # Every further call inside the window is a no-op: the timestamp does not
        # move, so no reduction over the state was performed.
        redirect_stderr(devnull) do
            for _ in 1:50
                FI.report_progress!(p, integ)
            end
        end
        @test p.t_wall_last == drawn_at
    end

    @testset "a nothing progress is a free no-op" begin
        sim = build_progress_sim(verbose = false)
        FI.init_problem!(sim)
        integ = FI.build_integrator(sim)
        @test FI.report_progress!(nothing, integ) === nothing
        @test FI.finish_progress!(nothing, integ) === nothing
        # With no bar running, per-output messages stay enabled...
        @test FI.verbose_log(build_progress_sim(verbose = true), nothing)
        # ...and are suppressed while one is, so `println`s cannot shred the bar.
        @test !FI.verbose_log(build_progress_sim(verbose = true),
            FI.ForwardProgress(build_progress_sim()))
    end

    @testset "run! is unaffected by whether the bar is on" begin
        quiet = build_progress_sim(verbose = false)
        run!(quiet)

        loud = build_progress_sim(verbose = true)
        redirect_stderr(devnull) do
            run!(loud)
        end
        @test loud.now.u == quiet.now.u
        @test loud.timer.t == quiet.timer.t
    end
end
