using CairoMakie
using FastIsostasy

n = 6
use_cuda = false
dense = false

T = Float64
W = T(3000e3)               # half-length of the square domain (m)
Omega = ComputationDomain(W, n, use_cuda = use_cuda, correct_distortion = false)
c = PhysicalConstants(rho_litho = 0.0)
p = LayeredEarth(Omega, layer_viscosities = [1e21], layer_boundaries = [88e3])

R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)

L_w = get_flexural_lengthscale(mean(p.litho_rigidity), c.rho_uppermantle, c.g)
kei = get_kei(Omega, L_w)
viscousgreen = calc_viscous_green(Omega, p, kei, L_w)
viscousconv = InplaceConvolution(viscousgreen, Omega.use_cuda)
u_equil = viscousconv(-Hcylinder * c.g * c.rho_ice)
heatmap(u_equil)