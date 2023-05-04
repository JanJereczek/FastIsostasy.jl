push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

n = 6
T = Float64
L = T(3000e3)               # half-length of the square domain (m)
Omega = ComputationDomain(L, n)
c = PhysicalConstants()
p = MultilayerEarth(Omega, c)

R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.(0.0:100.0:10_000.0)

t1 = time()
results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver="ExplicitEuler", active_geostate=false)
t_fastiso = time() - t1
println("Took $t_fastiso seconds!")
println("-------------------------------------")

Hice = [Hcylinder for t in t_out]
U = results.viscous
vo = init_optim(Omega, Hice, t_out, U)
opts = Options(x_abstol = 1e-2, x_reltol = 1e-2, g_abstol = 1e-3, g_reltol = 1e-3)

optim_results = optimize_viscosity(Omega, Hice, t_out, U, opts)