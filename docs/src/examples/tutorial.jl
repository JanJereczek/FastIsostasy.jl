#=
# Tutorial

In this section, we will present some examples with idealised loads and solid-Earth parameters. This should give the user a basic understanding of FastIsostasy's basic functions.

## 3D ➡ 2D Earth

FastIsostasy relies on a (polar) stereographic projection. Let's first create `Omega::ComputationDomain` and visualise how this relates to a domain on a spherical Earth:
=#

using CairoMakie, FastIsostasy

W = 3000e3      # (m) half-width of the domain Wx = Wy
n = 7           # implies an nx x ny grid with nx = ny = 2^n = 128.
Omega = ComputationDomain(W, n)
fig = Figure(size = (1600, 800), fontsize = 24)
ax1 = Axis3(fig[1, 1], title = "Original grid")
ax2 = Axis3(fig[1, 2], title = "Projected grid")
wireframe!(ax1, Omega.X .* Omega.K, Omega.Y .* Omega.K,
    Omega.R .* cos.(deg2rad.(Omega.Lat)), color = :gray10, linewidth = 0.1)
wireframe!(ax2, Omega.X .* Omega.K, Omega.Y .* Omega.K,
    Omega.null, color = :gray10, linewidth = 0.1)
for ax in [ax1, ax2]
    zlims!(ax, (0, 5e6))
    hidedecorations!(ax)
    hidespines!(ax)
end
fig

#=
!!! note "Using other projections"
    For now, FastIsostasy only supports the polar stereographic projection. Future releases will allow the user to define their own projection.

The projection allows to treat the radially-layered, onion-like structure of the solid Earth as a superposition of horizontal layers. Furthermore, FastIsostasy reduces this 3D problem into a 2D problem by collapsing the depth dimension, mainly through the computation of an effective viscosity field that accounts for the superposition of layers with different viscosities. The user is required to provide the 3D information, which will then be used under the hood to compute the effective viscosity. This tutorial shows such an example.

We want to render a situation similar to the one depicted below:

![Schematic representation of the three-layer set-up.](../assets/sketch_nlayer_model.png)

Initializing a [`LayeredEarth`](@ref) with parameters corresponding to this situation automatically computes the conversion from a 3D to a 2D problem. Since we will compare our solution to an analytical one of a flat Earth, we exceptionally switch off the distortion correction, which accounts for the distortion factor `Omega.K` in the computation. This can be simply executed by running:
=#

Omega = ComputationDomain(W, n, correct_distortion = false)
c = PhysicalConstants()
lv = [1e19, 1e21]       # viscosity layers (Pa s)
lb = [88e3, 400e3]      # depth of layer boundaries (m)
p = LayeredEarth(Omega, layer_viscosities = lv, layer_boundaries = lb, rho_litho = 0.0)
extrema(p.effective_viscosity)

#=
As expected, the effective viscosity is a homogeneous field. It corresponds to a nonlinear mean of the layered values provided by the user. Note that we have set $$\rho_{litho} = 0$$ to prevent the lithosphere from contributing to the hydrostatic pressure term, such that the numerical solution obtained here is comparable with the analytical one provided in [bueler-fast-2007](@citet). The default value however yields $$\rho_{litho} = 3200 \, \mathrm{kg/m^3}$$ and contributes to the hydrostatic pressure term.

The next section shows how to use the now obtained `p::LayeredEarth` for actual GIA computation.

## Simple load and geometry

We now apply a constant load, here a cylinder of ice with radius $$R = 1000 \, \mathrm{km}$$ and thickness $$H = 1 \, \mathrm{km}$$, over `Omega::ComputationDomain` introduced in [`LayeredEarth`](@ref). To formulate the problem conviniently, we use [`FastIsoProblem`](@ref), a struct containing the variables and options that are necessary to perform the integration over time. We can then simply apply `solve!(fip::FastIsoProblem)` to perform the integration of the ODE. Under the hood, the ODE is obtained from the PDE by applying a Fourier collocation scheme contained in [`lv_elva!`](@ref). The integration is performed according to `FastIsoProblem.diffeq::NamedTuple`, which contains the algorithm and optionally tolerances, maximum iteration number... etc.
=#

t_out = [0.0, 2e2, 6e2, 2e3, 5e3, 1e4, 5e4]     # vector of output time steps (yr)
εt = 1e-8
pushfirst!(t_out, -εt)                          # append step to have Heaviside at t=0

R = 1000e3                                      # ice disc radius (m)
H = 1e3                                         # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)   # field representing ice disk

t_Hice = [-εt, 0.0, t_out[end]]                 # ice history = Heaviside at t=0
Hice = [zeros(Omega.nx, Omega.ny), Hcylinder, Hcylinder]

fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, output = "sparse")
solve!(fip)

function plot3D(fip, k_idx)
    X, Y, out = Array(fip.Omega.X), Array(fip.Omega.Y), fip.out
    zl = extrema(out.ue[end] + out.u[end])
    fig = Figure(fontsize = 10)
    for j in eachindex(k_idx)
        ax = Axis3(fig[1, j])
        u_tot = out.ue[k_idx[j]] + out.u[k_idx[j]]
        surface!(ax, X, Y, u_tot, colormap = :cool)
        wireframe!(ax, X, Y, u_tot, color = :black, linewidth = 0.1)
        zlims!(ax, zl)
    end
    return fig
end
fig = plot3D(fip, [lastindex(t_out) ÷ 2, lastindex(t_out)])

#=
The figure above shows the total displacement at $$t = 0.6 \, \mathrm{kyr}$$ and $$t = 50 \, \mathrm{kyr}$$. Since we defined `output = "sparse"`, we can now access the elastic and viscous displacement at time `t_out[k]` by calling `fip.out.ue[k]` and `fip.out.u[k]`. For the present case, the latter can be compared to an analytic solution that is known for this particular case. Let's look at the accuracy of our numerical scheme over time by running following plotting commands:
=#

fig = Figure()
ax = Axis(fig[1, 1])
cmap = cgrad(:jet, length(t_out), categorical = true)
ii, jj = Omega.mx:Omega.nx, Omega.my
x = Omega.X[ii, jj]
r = Omega.R[ii, jj]

for k in eachindex(t_out)[2:end]
    analytic_solution_r(r) = analytic_solution(r, years2seconds(t_out[k]),
        c, p, H, R)
    u_analytic = analytic_solution_r.(r)
    u_numeric = fip.out.u[k][ii, jj]
    lines!(ax, x, u_analytic, color = cmap[k], linewidth = 5,
        label = L"$u_{ana}(t = %$(t_out[k]) \, \mathrm{yr})$")
    lines!(ax, x, u_numeric, color = cmap[k], linewidth = 5, linestyle = :dash,
        label = L"$u_{num}(t = %$(t_out[k]) \, \mathrm{yr})$")
end
axislegend(ax, position = :rb, patchsize = (50.0f0, 20.0f0))
fig

#=
## GPU support

For about $$n \geq 7$$, the present example can be computed even faster by using GPU parallelism. It could not represent less work from the user's perspective, as it boils down to calling [`ComputationDomain`](@ref) with an extra keyword argument:

```julia
Omega = ComputationDomain(W, n, use_cuda = true)
```

We then pass `Omega` to a `LayeredEarth` and a `FastIsoProblem`, which we solve as done above: that's it!

!!! info "Only CUDA supported!"
    For now only Nvidia GPUs are supported and there is no plan of extending this compatibility at this point.


## Make your own time loop

As any high-level function, [`solve!`](@ref) has some limitations. An ice-sheet modeller typically wants to embed FastIsostasy within a time-stepping loop. This can be easily done by getting familiar with some intermediate-level functions like [`init`](@ref), [`step!`](@ref) and [`write_out!`](@ref):
=#

Omega = ComputationDomain(3000e3, n)
p = LayeredEarth(Omega)
fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, output = "sparse")

update_diagnostics!(fip.now.dudt, fip.now.u, fip, 0.0)
write_out!(fip.nout, fip.now)
ode = init(fip)
@inbounds for k in eachindex(fip.out.t)[2:end]
    step!(fip, ode, (fip.out.t[k-1], fip.out.t[k]))
    write_out!(fip.nout, fip.now)
end
fig = plot3D(fip, [lastindex(t_out) ÷ 2, lastindex(t_out)])

#=
!!! warning "Limited GPU support"
    [`step!`](@ref) does not support GPU computation so far. Make sure your model is initialized
    on CPU.

## Using different deformation models

ELRA is a GIA model that is commonly used in ice-sheet modelling. For the vast majority of applications, it is less accurate than LV-ELVA without providing any significant speed up [swierczek2024fastisostasy](@cite). However, it can be used by specifying adequate options:
=#

p = LayeredEarth(Omega, tau = 3e3)
opts = SolverOptions(deformation_model = :elra)
fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts, output = "sparse")
solve!(fip)
fig = plot3D(fip, [lastindex(t_out) ÷ 2, lastindex(t_out)])

#=

The reader may have noticed that the equilibrium displacement (at $$t = 50 \, \mathrm{kyr}$$) given by ELRA is the same as the one given by LV-ELVA, although their transient behaviour differ (e.g. at $$t = 0.6 \, \mathrm{kyr}$$). This is expected when the lithosphere yields a constant thickness but breaks down when the lithosphere thickness varies.

=#