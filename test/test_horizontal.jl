using FastIsostasy
W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 7           # implies an nx x ny grid with nx = ny = 2^n = 128.
domain = RegionalDomain(W, n, correct_distortion = true)
c = PhysicalConstants()
layering = UniformLayering(2, [88e3, 400e3])
viscosities = [1e19, 1e21]                      # viscosity layers (Pa s)
p = SolidEarthParameters(domain, layer_viscosities = viscosities,
    layering = layering, rho_litho = 3.2e3)
extrema(p.effective_viscosity)

t_out = [0.0, 2e2, 6e2, 2e3, 5e3, 1e4, 5e4]     # vector of output time steps (yr)
εt = 1e-8
pushfirst!(t_out, -εt)                          # append step to have Heaviside at t=0

R = 1000e3                                      # ice disc radius (m)
H = 1e3                                         # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(domain, R, H)   # field representing ice disk

t_Hice = [-εt, 0.0, t_out[end]]                 # ice history = Heaviside at t=0
Hice = [zeros(domain.nx, domain.ny), Hcylinder, Hcylinder]

sim = Simulation(domain, c, p, t_out, t_Hice, Hice, output = "intermediate")
@time solve!(sim)

sim.ncout.computation_time # now gives about 11s vs. 20s in publication!

extrema(sim.now.u)

u_x, u_y = thinplate_horizontal_displacement(sim.now.u + sim.now.ue, p.litho_thickness, domain)
u_h = sqrt.(u_x.^2 .+ u_y.^2)
extrema(u_h)
# Very much in line with Spada (2011) fig 9.

lines(u_x[domain.mx:end, domain.my], color = :black, linewidth = 1)
lines(u_y[domain.mx:end, domain.my], color = :black, linewidth = 1)
s = 4
arrows(domain.x[1:s:end], domain.y[1:s:end], u_x[1:s:end, 1:s:end],
    u_y[1:s:end, 1:s:end], arrowsize = 3, lengthscale = 1e4,
    arrowcolor = :gray10, linecolor = :gray10)
heatmap(u_h)

sim = Simulation(domain, c, p, t_out, t_Hice, Hice, output = "sparse")
integrator = init_integrator(sim)
step!(integrator, t_out[end], true)
extrema(integrator.p.now.u)
extrema(sim.now.u)
heatmap(integrator.p.now.u)
# Works perfectly!!!