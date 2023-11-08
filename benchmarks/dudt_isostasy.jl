using FastIsostasy
W = 3000e3
n = 7
Omega = ComputationDomain(W, n, correct_distortion = false)
c = PhysicalConstants(rho_litho = 0.0)
lv = [1e19, 1e21]       # viscosity layers (Pa s)
lb = [88e3, 400e3]      # depth of layer boundaries (m)
p = LayeredEarth(Omega, layer_viscosities = lv, layer_boundaries = lb)

R = 1000e3                  # ice disc radius (m)
H = 1e3                     # ice disc thickness (m)
Hice = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.([0.0, 200.0, 600.0, 2000.0, 5000.0, 10_000.0, 50_000.0])
interactive_sealevel = false
fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hice)

u = copy(fip.out.u[1])
dudt = copy(fip.out.dudt[1])
t = 0.0

@code_warntype dudt_isostasy!(dudt, u, fip, t)

@btime dudt_isostasy!($dudt, $u, $fip, $t)
#=
Initial :               640.189 μs (39 allocations: 2.00 MiB)
fixed prealloc type:    621.213 μs (33 allocations: 2.00 MiB)
=#