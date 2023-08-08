# Examples

## Multi-layer Earth

FastIsostasy relies on a (polar) stereographic projection that allows to treat the radially-layered, onion-like structure of the solid Earth as a superposition of horizontal layers. Furthermore, FastIsostasy reduces this 3D problem into a 2D problem by collapsing the depth dimension, mainly through the computation of an effective viscosity field that accounts for the superposition of layers with different viscosities. The user is required to provide the 3D information, which will then be used under the hood to compute the effective viscosity. This tutorial shows such an example.

We want to render a situation similar to the one depicted below:

![Schematic representation of the three-layer set-up.](assets/sketch_nlayer_model.png)

Initializing a [`LateralVariability`](@ref) with parameters corresponding to this situation automatically computes the conversion from a 3D to a 2D problem. This can be simply executed by running:

```@example MAIN
using FastIsostasy

W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 6           # implies an Nx x Ny grid with Nx = Ny = 2^n = 64.
Omega = ComputationDomain(W, n)
c = PhysicalConstants(rho_litho = 0.0)

lv = [1e19, 1e21]       # (Pa s)
lb = [88e3, 400e3]      # (m)
p = LateralVariability(Omega, layer_viscosities = lv, layer_boundaries = lb)
extrema(p.effective_viscosity)
```

As expected, the effective viscosity is a homogeneous field. It corresponds to a nonlinear mean of the layered values provided by the user. The next section shows how to use the now obtained `p::LateralVariability` for actual GIA computation.

## Simple load and geometry

We now apply a constant load, here a cylinder of ice with radius ``R = 1000 \, \mathrm{km}`` and thickness ``H = 1 \, \mathrm{km}``, over the domain introduced in [Multi-layer Earth](@ref). To obtain the bedrock displacement over time and store it at time steps specified by a vector `t_out`, we can use the convenience function [`fastisostasy`](@ref) and the ODE solver `BS3()` from [OrdinaryDifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/):

```@example MAIN
using CairoMakie

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)
Hice = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.([0.0, 200.0, 600.0, 2000.0, 5000.0, 10_000.0, 50_000.0])

results = fastisostasy(t_out, Omega, c, p, Hice, ODEsolver = BS3())
function plot3D_fastiso_results(Omega, results)
    X, Y = Array(Omega.X), Array(Omega.Y)
    fig, ax, srf = surface(X, Y, results.ue_out[end] + results.u_out[end],
        axis=(type=Axis3,), colormap = :cool)
    wireframe!(ax, X, Y, results.ue_out[end] + results.u_out[end],
        color = :black, linewidth = 0.1)
    return fig
end
plot3D_fastiso_results(Omega, results)
```

... and here goes the total displacement at ``t = 50 \, \mathrm{kyr}``! You can now access the elastic and viscous displacement at time `t_out[k]` by calling `results.ue_out[k]` or `results.u_out[k]`. For the present case, the latter can be compared to an analytic solution that is known for this particular case. Let's look at the accuracy of our numerical scheme over time by running following plotting commands:

```@example MAIN
ii, jj = Omega.Mx:Omega.Nx, Omega.My
x = Omega.X[ii, jj]
r = Omega.R[ii, jj]

fig = Figure()
ax = Axis(fig[1, 1])
cmap = cgrad(:jet, length(t_out), categorical = true)
analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)

for i in eachindex(t_out)
    analytic_solution_r(r) = analytic_solution(r, t_out[i], c, p, H, R, analytic_support)
    u_analytic = analytic_solution_r.(r)
    u_numeric = results.u_out[i][ii, jj]
    lines!(ax, x, u_analytic, color = cmap[i], linewidth = 5,
        label = L"$u_{ana}(t = %$(round(seconds2years(t_out[i]))) \, \mathrm{yr})$")
    lines!(ax, x, u_numeric, color = cmap[i], linewidth = 5, linestyle = :dash,
        label = L"$u_{num}(t = %$(round(seconds2years(t_out[i]))) \, \mathrm{yr})$")
end
axislegend(ax, position = :rb, nbanks = 2, patchsize = (50.0f0, 20.0f0))
fig
```

That looks pretty good! One might however object that the convenience function [`fastisostasy`](@ref) ends up being not so convenient as soon as the ice load changes over time. This case can however be easily handled, as shown in the next section.
### Time-varying load

By providing snapshots of the ice thickness and their associated time to [`fastisostasy`](@ref), an interpolator is created and called within the time integration. Let's create a tool example where the thickness of the ice cylinder asymptotically grows from 0 to 1 km with a relaxation time of ``\tau = 50 \, \mathrm{kyr}``. Thus, the load applied at the end of the simulation is smaller than what was observed in the previous example, resulting in smaller maximal displacement. However, the shape of the bedrock should be similar, since the isostatic response closely follows equilibrium for such large time scales of the load. This can be verified by running:

```@example MAIN
tau = years2seconds(50000.0)
normalized_asymptote(t) = 1 - exp(-t/tau)
t_asymptotic = collect(0.0:years2seconds(5000.0):t_out[end])

H_asymptotic = [Hice .* normalized_asymptote(t) for t in t_asymptotic]
# results_asymptotic = fastisostasy(t_out, Omega, c, p, t_asymptotic, H_asymptotic,
#     ODEsolver = "ExplicitEuler")
# plot3D_fastiso_results(Omega, results_asymptotic)
```
### GPU support

For about $n > 6$, the previous example can be computed even faster by using GPU parallelism. It could not represent less work from the user's perspective, as it boils down to calling the `ComputationDomain` with an extra keyword argument and passing it to a `::LateralVariability` with the viscosity and depth values defined earlier:

```@example MAIN
n = 7
Omega = ComputationDomain(W, n, use_cuda = true)
p = LateralVariability(Omega, layer_viscosities = lv, layer_boundaries = lb)
Hice = uniform_ice_cylinder(Omega, R, H)

#results_gpu = fastisostasy(t_out, Omega, c, p, Hice, ODEsolver = BS3())
#plot3D_fastiso_results(Omega, results_gpu)
```

That's it, nothing more!

!!! info "Only CUDA supported!"
    For now only Nvidia GPUs are supported and there is no plan of extending this compatibility at this point.

## Make your own time loop

As any high-level function, [`fastisostasy`](@ref) has limitations. An ice-sheet modeller typically wants to embed FastIsostasy within a time-stepping loop. This can be easily done by getting familiar with some intermediate-level functions:

```@example MAIN
Omega = ComputationDomain(3000e3, 6)
c = PhysicalConstants(rho_litho = 0.0)
p = LateralVariability(Omega)
Hice = uniform_ice_cylinder(Omega, R, H)

interactive_geostate = false
fi = FastIso(Omega, c, p, t_out, interactive_geostate)
dt = years2seconds(1.0)
t = t_out[1]:dt:t_out[end]

for k in eachindex(t)
    if minimum(abs.(t[k] .- t_out)) < years2seconds(0.1)
        println("t = $(round(seconds2years(t[k]), sigdigits=1)) yr,    "*
                "u_min = $(round(minimum(fi.geostate.u), digits=2)) m")
    end
    update_loadcolumns!(fi, Hice)
    update_diagnostics!(fi.geostate.dudt, fi.geostate.u, fi, t[k])
    explicit_euler!(fi.geostate.u, fi.geostate.dudt, dt)
end
```

Contrary to the previous examples, an explicit Euler method is used for integration. Whereas the previously used solvers were part of OrdinaryDiffEq.jl, the current one is a lightweigth implementation that aims to avoid a `remake` of the `ODEProblem` at each iteration. The latter option can however be a good idea **if you can afford large coupling time steps**. Here is an example of it:

```@example MAIN
dt_couple = years2seconds(200.0)
t = collect(t_out[1]:dt_couple:t_out[end])
fi = FastIso(Omega, c, p, t, interactive_geostate)
prob0 = ODEProblem(update_diagnostics!, copy(fi.geostate.u), (t[1], t[2]), fi)

for k in eachindex(t)[2:end]
    update_loadcolumns!(fi, Hice)
    prob = remake(prob0, u0 = fi.geostate.u, tspan = (t[k-1], t[k]), p = fi)
    sol = solve(prob, BS3(), reltol=1e-3)
    if minimum(abs.(t[k] .- t_out)) < years2seconds(0.1)
        println("t = $(round(seconds2years(t[k]), sigdigits=1)) years,    "*
                "u_min = $(round(minimum(fi.geostate.u), digits=2)) meters")
    end
end
```

!!! info "Coupling to julia Ice-Sheet model"
    In case your Ice-Sheet model is programmed in julia, we highly recommend performing
    the coupling within the function updating the derivatives and let `OrdinaryDiffEq.jl`
    handle the rest.

## Antarctic deglaciation

We now want to provide a tough example that presents:
- a heterogeneous lithosphere thickness
- a heterogeneous upper-mantle viscosity
- various viscous channels
- a more elaborate load that evolves over time
- changes in the sea-level

For this we run a deglaciation of Antarctica, based on the ice thickness estimated in [GLAC1D]().

```@example MAIN
W = 3000e3      # (m) half-width of the domain
n = 7           # implies an NxN grid with N = 2^n = 128.
Omega = ComputationDomain(W, n)
c = PhysicalConstants()
```

## Inversion of viscosity field

FastIsostasy.jl relies on simplification of the full problem and might therefore need a calibration step to match real data or 3D GIA model output, thereafter simply referred to by data. By means of an unscented Kalman inversion, one can e.g. infer the appropriate field of effective mantle viscosity that matches the data best. Whereas this is known to be a tedious step, FastIsostasy is developped to ease the procedure by providing a convenience struct `Paraminversion` that can be run by:

```@example MAIN
Omega = ComputationDomain(W, n)

lb = [88e3, 180e3, 280e3, 400e3]
lv = get_wiens_layervisc(Omega)
p = LateralVariability(Omega, layer_boundaries = lb, layer_viscosities = lv)
ground_truth = copy(p.effective_viscosity)

R = T(2000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.(0.0:1_000.0:2_000.0)

results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3(), interactive_geostate=false)

# tinv = t_out[2:end]
# Hice = [Hcylinder for t in tinv]
# Y = results.u_out[2:end]
# paraminv = ParamInversion(Omega, c, p, tinv, Y, Hice)
# priors, ukiobj = perform(paraminv)
# logeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, paraminv)
```