include("helpers.jl")

fip = minimal_fip()
u = copy(fip.out.u[1])
dudt = copy(fip.out.dudt[1])
t = 0.0

@code_warntype lv_elva!(dudt, u, fip, t)

@btime lv_elva!($dudt, $u, $fip, $t)
#=
Initial :               640.189 μs (39 allocations: 2.00 MiB)
fixed prealloc type:    621.213 μs (33 allocations: 2.00 MiB)
=#