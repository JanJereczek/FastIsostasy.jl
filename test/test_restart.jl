# Restart files (src/restart.jl): stopping a run, writing its state to NetCDF and
# resuming from that file must continue the trajectory of an uninterrupted run.
# The only expected difference is the adaptive step size, which the restarted
# integrator re-estimates, hence the tolerance rather than bitwise equality.

using FastIsostasy
using Test

const FI = FastIsostasy

const T_UP, T_DOWN, T_END = 10f3, 20f3, 30f3

# Triangular load: ice grows from 0 to 10 kyr, melts from 10 to 20 kyr, and stays
# absent until 30 kyr.
function build_restart_sim(case, t_span; n = 5, kwargs...)
    domain = RegionalDomain(3f6, n)
    H0 = zeros(domain)
    H1 = 1f3 .* (domain.R .< 1f6)
    it = TimeInterpolatedIceThickness([0f0, T_UP, T_DOWN, T_END],
        [H0, H1, H0, H0], domain)
    bcs = BoundaryConditions(domain, ice_thickness = it)
    if case == :maxwell
        opts = SolverOptions(show_progress = false)
        se = SolidEarth(domain; layer_boundaries = [88f3], layer_viscosities = [1f21])
        return Simulation(domain, bcs, RegionalSeaLevel(), se, t_span;
            opts = opts, kwargs...)
    else
        # Transient creep (Kelvin-branch state `u_K`), the kinematic BSL formalism
        # (two-time-level buffers) and an interactive sea-level load on a marine
        # margin, so that every piece of state a restart must carry is non-trivial.
        # The semi-implicit Kelvin update needs a fixed step.
        opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = 10f0))
        mantle = TransientCreepMantle(shearmodulus = 67e9,
            relaxation_strength = 1.2, kelvin_time = 500.0)
        se = SolidEarth(domain; mantle = mantle, maskactive = domain.R .< 1.5f6,
            lithosphere = LaterallyConstantLithosphere(),
            layer_boundaries = [88f3], layer_viscosities = [1f21])
        sealevel = RegionalSeaLevel(load = InteractiveSealevelLoad(),
            bsl = PiecewiseConstantBSL(), formalism = AdhikariBSLFormalism())
        z_b_ref = ifelse.(domain.R .< 8f5, 200f0, -300f0)
        return Simulation(domain, bcs, sealevel, se, t_span;
            opts = opts, z_b_ref = z_b_ref, kwargs...)
    end
end

const RESTART_VARS = [:u, :ue, :z_b, :z_ss]

@testset "restart files" begin
    for case in (:maxwell, :transient_kinematic)
        @testset "$case" begin
            t_out = collect(T_UP:1f3:T_END)
            ref = build_restart_sim(case, (0f0, T_END);
                nout = NativeOutput(t = t_out, vars = RESTART_VARS))
            run!(ref)

            mktempdir() do dir
                path = joinpath(dir, "restart.nc")

                # First leg: load up to 10 kyr, writing a restart file at the end.
                leg1 = build_restart_sim(case, (0f0, T_UP); restartout = path)
                run!(leg1)
                @test isfile(path)
                @test !isfile(path * ".tmp")

                # Second leg: resume from the file until 30 kyr.
                leg2 = build_restart_sim(case, (T_UP, T_END); restart_from = path,
                    nout = NativeOutput(t = t_out, vars = RESTART_VARS))

                # The file round-trips every saved value exactly.
                for (name, owner, f) in FI.restart_leaves(leg1)
                    v1 = getfield(owner, f)
                    _, owner2, _ = only(filter(l -> l[1] == name,
                        FI.restart_leaves(leg2)))
                    @test getfield(owner2, f) == v1
                end
                @test leg2.timer.t == T_UP
                @test leg2.timer.t_sparse0 == 0f0

                run!(leg2)

                # The transient of the restarted run matches the uninterrupted one.
                # The error is measured against the peak of each field over the
                # transient: a field passing through zero (e.g. the elastic
                # displacement when the ice is gone at 20 kyr) has no meaningful
                # pointwise relative error.
                for var in RESTART_VARS
                    scale = maximum(maximum(abs, x) for x in ref.nout.vals[var])
                    for k in eachindex(t_out)
                        @test maximum(abs,
                            leg2.nout.vals[var][k] .- ref.nout.vals[var][k]) <=
                              1f-3 * scale
                    end
                end
                @test isapprox(leg2.sealevel.bsl.z, ref.sealevel.bsl.z;
                    rtol = 1f-3, atol = 1f-6)
            end
        end
    end

    @testset "mismatches are rejected" begin
        mktempdir() do dir
            path = joinpath(dir, "restart.nc")
            sim = build_restart_sim(:maxwell, (0f0, 1f3))
            run!(sim)
            write_restart(path, sim)

            # wrong start time
            @test_throws ErrorException build_restart_sim(:maxwell, (0f0, 2f3);
                restart_from = path)
            # wrong grid
            @test_throws ErrorException build_restart_sim(:maxwell, (1f3, 2f3);
                n = 4, restart_from = path)
            # physics carrying state the file does not have
            @test_throws ErrorException build_restart_sim(:transient_kinematic,
                (1f3, 2f3); restart_from = path)
            # not a restart file
            bogus = joinpath(dir, "bogus.nc")
            write(bogus, "not netcdf")
            @test_throws Exception read_restart!(sim, bogus)
        end
    end

    @testset "intermediate restart times" begin
        mktempdir() do dir
            path = joinpath(dir, "restart.nc")
            ro = RestartOutput(path; t = [5f2, 1f3, 1.5f3])
            sim = build_restart_sim(:maxwell, (0f0, 1f3); restartout = ro)
            @test sim.restartout.k == 1
            run!(sim)
            # 1.5 kyr lies beyond the run; 0.5 and 1 kyr were written
            @test sim.restartout.k == 3
            @test isfile(path)
            resumed = build_restart_sim(:maxwell, (1f3, 2f3); restart_from = path,
                restartout = ro)
            # times up to the restart are skipped
            @test resumed.restartout.k == 3
        end
    end
end
