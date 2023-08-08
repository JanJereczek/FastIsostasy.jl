using FastIsostasy
using CairoMakie

W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 6           # implies an Nx x Ny grid with Nx = Ny = 2^n = 64.
Omega = ComputationDomain(W, n, use_cuda = false)
c = PhysicalConstants()

lv = [1e19, 1e21]       # (Pa s)
lb = [88e3, 400e3]      # (m)
p = LateralVariability(Omega, layer_viscosities = lv, layer_boundaries = lb)

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)
Hice = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
results_gpu = fastisostasy(t_out, Omega, c, p, Hice, ODEsolver = BS3())
