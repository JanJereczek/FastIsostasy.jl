using FastIsostasy

W, n = 3f6, 6
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
    viscous_displacement = DistanceWeightedBC(RegularBCSpace(), 0f0),
    elastic_displacement = BorderBC(ExtendedBCSpace(), 0f0),
)
sem = SolidEarthModel(
    LaterallyVariableLithosphere(),
    MaxwellMantle(),
)
sep = SolidEarthParameters(Omega, rho_litho = 0f0)
nout = NativeOutput(vars = [:u, :ue, :dz_ss, :H_ice, :u_x, :u_y], t = t_ice)
fip = FastIsoProblem(Omega, sem, sep; bcs = bcs, nout = nout)
@time solve!(fip)
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
@btime update_diagnostics!($dudt, $u, $fip, $t)
# n = 6: 98.118 μs (11 allocations: 96.35 KiB)
# After fixing interpolation: n = 6: 91.041 μs (2 allocations: 96 bytes)
# After fixing mem allocation from bcs: 86.605 μs (0 allocations: 0 bytes)
# After optimizing each step: 61.808 μs (0 allocations: 0 bytes)
# n = 7: 495.489 μs (11 allocations: 384.35 KiB)
# n = 7, after fixing mem allocs and optimizing each step: 346.402 μs (0 allocations: 0 bytes)

# Since profview is not proviing legible output (to me), let's use @btime:
# n = 6
@btime apply_bc!($u, $fip.bcs.viscous_displacement)
# 287.883 ns (0 allocations: 0 bytes)
@btime apply_bc!($fip.now.H_ice, $t, $fip.bcs.ice_thickness)
# 849.536 ns (0 allocations: 0 bytes)
@btime update_Haf!($fip)
# no mul with rho_sw_ice: 938.321 ns (0 allocations: 0 bytes)
# mul with rho_sw_ice in same expression: 9.766 μs (0 allocations: 0 bytes)
# mul with rho_sw_ice in next line: 1.124 μs (0 allocations: 0 bytes)
@btime columnanom_load!($fip)
# one liner (previous version): about 20 μs
# decomposed into various column anomalies: 1.751 μs (0 allocations: 0 bytes)
lc = LaterallyConstantLithosphere()
@btime update_elasticresponse!($fip, $lc)
# 674.266 μs (0 allocations: 0 bytes) => huge bottleneck!
# After in place planned convolution: 109.584 μs (0 allocations: 0 bytes)

@btime columnanom_litho!($fip)
# 516.802 ns (0 allocations: 0 bytes)
@btime update_bsl!($fip)
# 19.089 μs (15 allocations: 80.43 KiB)
@btime update_V_af!($fip)
# 2.477 μs (6 allocations: 32.17 KiB)
# using buffer:   1.013 μs (0 allocations: 0 bytes)
@btime update_V_pov!($fip)
# 15.095 μs (6 allocations: 32.17 KiB)
# using buffer: 6.391 μs (0 allocations: 0 bytes)
@btime update_V_den!($fip)
# 1.423 μs (3 allocations: 16.09 KiB)
# using buffer:   938.154 ns (0 allocations: 0 bytes)
@btime total_volume($fip)
# @btime update_sealevel!($fip, $fip.bcs.z_ss)
@btime columnanom_full!($fip)
#   719.178 ns (0 allocations: 0 bytes)
@btime update_dudt!($dudt, $u, $fip, $t, $fip.em)
@btime columnanom_mantle!($fip)
@btime update_bedrock!($fip, $u)

using Profile
@profview_allocs update_diagnostics!(dudt, u, fip, t) sample_rate=0.1

@profview update_diagnostics!(dudt, u, fip, t)
# @time update_diagnostics!(dudt, u, fip, t)
# @code_warntype update_diagnostics!(dudt, u, fip, t)

# Problem:
@btime update_ice!($fip.now.H_ice, $t, $fip.it)
# n = 6: 6.839 μs (9 allocations: 96.26 KiB)

# Solution:
itp = TimeInterpolation2D(fip.it.t_vec, fip.it.H_vec)
interpolate!(fip.now.H_ice, 1f0, itp)
@btime interpolate!($fip.now.H_ice, $1f0, $itp)
# n = 6: 183.162 ns (0 allocations: 0 bytes)