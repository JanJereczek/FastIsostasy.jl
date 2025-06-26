using FastIsostasy

n, W = 6, 3f6
use_cuda = false
Omega = RegionalComputationDomain(W, n, use_cuda = use_cuda)

H_ice_0 = kernelnull(Omega)
H_ice_1 = 1f3 .* (Omega.R .< 1f6)
t_ice = [0, 1, 50f3]
H_ice = [H_ice_0, H_ice_1, H_ice_1]
it = TimeInterpolatedIceThickness(t_ice, H_ice, Omega)

bcs = ProblemBCs(
    Omega,
    ice_thickness = it,
    sea_level = ConstantSeaLevel(),
    viscous_displacement = DistanceWeightedBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
    geoid_perturbation = BorderBC(ExtendedBCSpace(), 0f0),
)
em = EarthModel(
    RigidLithosphere(),
    LaterallyVariableMantle(),
    MaxwellRheology(),
)
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y], t = t_ice)
p = LayeredEarth(Omega, rho_litho = 0f0)
opts = SolverOptions(diffeq = DiffEqOptions(alg = Heun()))

fip = FastIsoProblem(Omega, em, p; bcs = bcs, opts = opts, nout = nout)

solve!(fip)
println("Computation time: ", fip.nout.computation_time)
# Computation time: 62.86226 ==> pretty bad compared to publication. Due to type specification issues?
# n = 6, 0 mem alloc, Computation time: 12.50684 s
# n = 6, gpu, 0 mem alloc, Computation time: 55.2 s
# Ok, looks like it's not the broadcast (and not even update_diagnostics!)...
# Maybe time step extremely small?
# Yes!!! Mean time step is 0.5 year which is too small imo. Possible reasons:
# - BCs are not very smooth in time
# - 4th order derivatives are noisy
# - Coupled to elastic, n = 6,
    # RigidLithosphere: 9.006247 s

dudt = fip.now.dudt
u = fip.now.u
t = 10.0f0
update_diagnostics!(dudt, u, fip, t)
# @time update_diagnostics!(dudt, u, fip, t)
# @code_warntype update_diagnostics!(dudt, u, fip, t)

# @btime update_diagnostics!($dudt, $u, $fip, $t)
# n = 6: 98.118 μs (11 allocations: 96.35 KiB)
# After fixing interpolation: n = 6: 91.041 μs (2 allocations: 96 bytes)
# After fixing mem allocation from bcs: 86.605 μs (0 allocations: 0 bytes)
# n = 7: 495.489 μs (11 allocations: 384.35 KiB)


using Profile
@profview_allocs update_diagnostics!(dudt, u, fip, t) sample_rate=0.1


# Problem:
@btime update_ice!($fip.now.H_ice, $t, $fip.it)
# n = 6: 6.839 μs (9 allocations: 96.26 KiB)

# Solution:
itp = TimeInterpolation2D(fip.it.t_vec, fip.it.H_vec)
interpolate!(fip.now.H_ice, 1f0, itp)
@btime interpolate!($fip.now.H_ice, $1f0, $itp)
# n = 6: 183.162 ns (0 allocations: 0 bytes)