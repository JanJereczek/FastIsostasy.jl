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
t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

results = fastisostasy(t_out, Omega, c, p, Hice, ODEsolver = BS3())
function plot3D_fastiso_results(Omega, results)
    fig, ax, srf = surface(Omega.X, Omega.Y, results.ue_out[end] + results.u_out[end],
        axis=(type=Axis3,), colormap = :cool)
    wireframe!(ax, Omega.X, Omega.Y, results.ue_out[end] + results.u_out[end],
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
results_asymptotic = fastisostasy(t_out, Omega, c, p, t_asymptotic, H_asymptotic,
    ODEsolver = BS3())
plot3D_fastiso_results(Omega, results_asymptotic)
```
### GPU support

For about $n > 6$, the previous example can be computed even faster by using GPU parallelism. It could not represent less work from the user's perspective, as it boils down to calling the `ComputationDomain` with an extra keyword argument and passing it to a `::LateralVariability` with the viscosity and depth values defined earlier:

```@example MAIN
n = 7
Omega = ComputationDomain(W, n, use_cuda = true)
p = LateralVariability(Omega, layer_viscosities = lv, layer_boundaries = lb)
Hice = uniform_ice_cylinder(Omega, R, H)

results_gpu = fastisostasy(t_out, Omega, c, p, Hice, ODEsolver = BS3())
plot3D_fastiso_results(Omega, results_gpu)
```

That's it, nothing more!

!!! info "Only CUDA supported!"
    For now only Nvidia GPUs are supported and there is no plan of extending this compatibility at this point.

## Simple load and geometry - DIY

Nonetheless, as any high-level convenience function, [`fastisostasy`](@ref) has limitations. An ice-sheet modeller typically wants to embed FastIsostasy within a time-stepping loop. This can be easily done by getting familiar with some intermediate-level functions. We here illustrate this by letting an ice cap grow over time. This growth is unphysical for the sake of keeping the example simple. 

```@example MAIN
W = 3000e3
n = 6
Omega = ComputationDomain(W, n)
c = PhysicalConstants()
p = LateralVariability(Omega)

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)

u_0, ue_0 = copy(Omega.null), copy(Omega.null)
fi = FastIso(Omega, c, p, t_Hice_snapshots, Hice_snapshots,
    t_eta_snapshots, eta_snapshots, interactive_geostate; kwargs...)
u = copy(u_0)

for t in 0.0:10.0:100.0
    fi.Hice = Hice .* normalized_asymptote(t)
    u, dudt, ue, geoid, sealevel = forward_isostasy(dt, t_out, u, fi, BS3(), false)
    println("t = $t,    u_max = $(maximum(u)),    dudt_max = $(maximum(dudt))")
end
```