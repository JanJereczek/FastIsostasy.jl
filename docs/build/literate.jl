using FastIsostasy
# using CairoMakie

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
results_gpu = fastisostasy(t_out, Omega, c, p, Hice, alg = BS3())


#####################################################

W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 6
Omega = ComputationDomain(W, n)
c = PhysicalConstants()
lv = [1e19, 1e21]       # (Pa s)
lb = [88e3, 400e3]      # (m)
p = LateralVariability(Omega, layer_viscosities = lv, layer_boundaries = lb)

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
p = LateralVariability(Omega, layer_viscosities = [1e19, 1e21], layer_boundaries = [88e3, 400e3])

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











using FastIsostasy
Omega = ComputationDomain(3000e3, 6)
lb = [88e3, 180e3, 280e3, 400e3]
lv = get_wiens_layervisc(Omega)
p = LateralVariability(Omega, layer_boundaries = lb, layer_viscosities = lv)
R, H = 2000e3, 1e3
Hcylinder = uniform_ice_cylinder(Omega, R, H)
Hice = [Hcylinder for t in t_out]
t_out = years2seconds.(1_000.0:1_000.0:2_000.0)
fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hice)
solve!(fip)
ground_truth = copy(p.effective_viscosity)

config = InversionConfig()
data = InversionData(t_out, fip.out.u, Hice, config)
paraminv = ParamInversion(Omega, c, p, config, data)
priors, ukiobj = perform(paraminv)
logeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, paraminv)