using FastIsostasy
# using CairoMakie

W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 6           # implies an Nx x Ny grid with Nx = Ny = 2^n = 64.
Omega = ComputationDomain(W, n, use_cuda = false)
c = PhysicalConstants()

lv = [1e19, 1e21]       # (Pa s)
lb = [88e3, 400e3]      # (m)
p = LayeredEarth(Omega, layer_viscosities = lv, layer_boundaries = lb)

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)
Hice = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
results_gpu = fastisostasy(t_out, Omega, c, p, Hice, alg = BS3())


#####################################################

W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 6
Omega = ComputationDomain(W, n)
c = PhysicalConstants()
lv = [1e19, 1e21]       # (Pa s)
lb = [88e3, 400e3]      # (m)
p = LayeredEarth(Omega, layer_viscosities = lv, layer_boundaries = lb)

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)
Hice = uniform_ice_cylinder(Omega, R, H)

interactive_sealevel, verbose = false, true
t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel)
dt_loop = years2seconds(100.0)
t_loop = t_out[1]:dt_loop:t_out[end]
dt = years2seconds(1.0)

tau = years2seconds(50000.0)
normalized_asymptote(t) = 1 - exp(-t/tau)

for k in eachindex(t_loop)[1:end-1]
    tvec = [t_loop[k], t_loop[k+1]]
    tmean = sum(tvec) / length(tvec)
    update_loadcolumns!(fip, Hice .* normalized_asymptote(tmean))
    forward_isostasy!(fip, tvec, dt, BS3(), verbose)
    t, u_min = round(seconds2years(t_loop[k+1])), minimum(fip.geostate.u)

    if minimum(abs.(t_loop[k+1] .- t_out)) < years2seconds(0.1)
        println("t = $t,    u_min = $u_min")
    end
end

#####################################################

Omega = ComputationDomain(3000e3, 6)
c = PhysicalConstants(rho_litho = 0.0)
p = LayeredEarth(Omega, layer_viscosities = [1e19, 1e21], layer_boundaries = [88e3, 400e3])

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)
Hice = uniform_ice_cylinder(Omega, R, H)
tau = years2seconds(50000.0)
normalized_asymptote(t) = 1 - exp(-t/tau)

interactive_sealevel, verbose = false, true
t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel)
dt = years2seconds(1.0)
t = t_out[1]:dt:t_out[end]

for k in eachindex(t)
    if minimum(abs.(t[k] .- t_out)) < years2seconds(0.1)
        println("t = $(round(seconds2years(t[k]), sigdigits=1)) years,    "*
                "u_min = $(round(minimum(fip.geostate.u), digits=2)) meters")
    end
    update_loadcolumns!(fip, Hice)   #  .* normalized_asymptote(t[k])
    update_diagnostics!(fip.geostate.dudt, fip.geostate.u, fip, t[k])
    simple_euler!(fip.geostate.u, fip.geostate.dudt, dt)
end



for k in eachindex(t)
    if minimum(abs.(t[k] .- t_out)) < years2seconds(0.1)
        println("t = $(round(seconds2years(t[k]), sigdigits=1)) years,    "*
                "u_min = $(round(minimum(fip.geostate.u), digits=2)) meters")
    end
    update_loadcolumns!(fip, Hice)   #  .* normalized_asymptote(t[k])
    update_diagnostics!(fip.geostate.dudt, fip.geostate.u, fip, t[k])
    explicit_rk4!(fip, dudt_isostasy!, dt, t[k])
end











using FastIsostasy, CairoMakie
Omega = ComputationDomain(3000e3, 5)
c = PhysicalConstants()
lb = [88e3, 180e3, 280e3, 400e3]
lv = load_wiens2021(Omega)
p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
R, H = 1000e3, 1e3
Hice = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.(1e3:1e3:2e3)
fip = FastIsoProblem(Omega, c, p, t_out, false, Hice)
solve!(fip)
ground_truth = copy(p.effective_viscosity)

config = InversionConfig()
data = InversionData(fip.out.t, fip.out.u, [Hice, Hice, Hice], config)
paraminv = InversionProblem(Omega, c, p, config, data)
priors, ukiobj = solve(paraminv)
logeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, data)