#=
# Parameter inversion

Assume a dynamical system $f$ generates an output $y$ based on its input $x$ and on a set of parameters $\theta$. The goal of parameter inversion is to estimate the parameters $\theta$ that best fit the data $\hat{y}$ with respect to an error metric $\Psi$, i.e. to solve the following optimization problem:

```math
y = f(x, \theta)
\hat{\theta} = \arg\min_{\theta} \Psi(\hat{y} - y)
```

Since FastIsostasy relies on simplifications of the full GIA problem, applying such inversion can be useful to tune the model parameters (upper-mantle viscosity field, lithospheric thickness... etc.) such that the output $y$ (typically the displacement) matches the data $\hat{y}$ (typically obtained from observations or from a 3D GIA model). There are many ways of solving this problem, among which the use of Kalman filtering techniques. FastIsostasy.jl provides convenience functions for the latter by wrapping the functionalities of EnsembleKalmanProcesses.jl in an external routine that is automatically loaded when `using EnsembleKalmanProcesses`.

We demonstrate the tuning of the effective viscosity in a region that is being forced by a circular ice load. We emphasise that the highlighted tools provided by FastIsostasy are not limited to this case. As it will be made clear throughout the example, the user merely needs to define their own `ParameterReduction` and the associated behaviours of `FastIsostasy.reconstruct!` and `FastIsostasy.extract_output` to adapt the inversion procedure to their needs (e.g. tune lithospheric thickness throughout the whole domain).

!!! note "Resolution"
    We perform the following analysis on a low-resolution grid. high resolutions (ca. ? `Omega.Nx = Omega.Ny > 200`) are difficult to achieve since the underlying unscented Kalman filter requires many simulations. This is however typically not a problem, since the parametric fields (here the viscosity) are smooth and can be downsampled without significant loss of information.

We first load the necessary packages, initialize the `ComputationDomain` and assign laterally-variable viscosity profiles to `p::LayeredEarth` by loading the fields estimated in [wiens-seismic-2022](@cite):
=#

using CairoMakie
using Distributions
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using FastIsostasy
using LinearAlgebra

Omega = ComputationDomain(3000e3, 5)
c = PhysicalConstants()
lb = [100e3, 200e3, 300e3]
_, eta, eta_itp = load_dataset("Wiens2022")
loglv = cat([eta_itp.(Omega.X, Omega.Y, z) for z in lb]..., dims = 3)
lv = 10 .^ loglv
p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
heatmap(log10.(p.effective_viscosity))

#=
To make this problem more exciting, we place the center of the ice load to $$(-1000, -1000) \: \mathrm{km}$$ where the viscosity field displays a less uniform structure. For the sake of simplicity, the data to fit is obtained from a FastIsostasy simulation with the ground-truth viscosity field.
=#

R, H = 1000e3, 1e3
Hcylinder = uniform_ice_cylinder(Omega, R, H, center = [-1000e3, -1000e3])
Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]
t_out = years2seconds.(collect(1e3:1e3:2e3))
pushfirst!(t_out, t_out[1]-1e-8)
t_Hice = copy(t_out)

true_viscosity = copy(p.effective_viscosity)
fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, output = "sparse")
solve!(fip)

#=
Assuming the result of this simulation to be the ground truth $\hat{y}$, we can now build an `InversionData` object that will be passed to an `InversionProblem`. The `InversionData` object contains the time series of the input field `X = Hice` and the output field `Y = extract_fip(fip)` (here the displacement field). Furthermore, we define an `InversionConfig` that uses the unscented Kalman filter as introduced in [huang-unscented-2021](@cite). 
=#

X = Hice

# Only tune viscosity in the region where there is ice.
mask3D = cat([x .> 0 for x in X]..., dims = 3)
mask = reduce(|, mask3D, dims = 3)[:, :, 1]

# Only use last time step for inversion
t_inv = t_out[end:end]
extract_last_viscous_displacement(fip) = fip.out.u[end:end]
extract_fip = extract_last_viscous_displacement
Y = extract_fip(fip)

data = InversionData(t_inv, length(t_inv), X, Y, mask, count(mask))
config = InversionConfig(Unscented, N_iter = 5)

#=
Before finalising the inversion, we need to specify a `ParameterReduction`, which allows a multiple dispatch of `reconstruct!`, `extract_output` and (optionally) `print_inversion_evolution`. This allows the inversion procedure to update the parameters, extract the relevant output and (optionally) print out meaningful information over the iterations of the inversion procedure. In this example, we define a `ViscosityRegion` that reduces the number of parameters to the number of grid points in the mask. The `reconstruct!` function updates the effective viscosity field with the parameters, and the `extract_output` function extracts the displacement field from the output of the FastIsostasy simulation.
=#

struct ViscosityRegion <: ParameterReduction
    mask::BitMatrix
    nparams::Int
end

function FastIsostasy.reconstruct!(fip, params, reduction::ViscosityRegion)
    fip.p.effective_viscosity[reduction.mask] .= 10 .^ params
end

function FastIsostasy.extract_output(fip, reduction::ViscosityRegion)
    return extract_fip(fip)[1][reduction.mask]
end

function FastIsostasy.print_inversion_evolution(paraminv, n, ϕ_n, reduction::ViscosityRegion)
    err_n = paraminv.error[n]
    cov_n = paraminv.ukiobj.process.uu_cov[n]

    println("-----------------------------------------------------")
    println("Inversion being performed with $(typeof(reduction)) as a parameter reduction")
    @show size(ϕ_n), n, err_n, norm(cov_n)
    println("Mean tuned viscosity: $(round(mean(ϕ_n), digits = 3))")
    println("Extrema of tuned viscosity: $(extrema(ϕ_n))")
    return nothing
end

reduction = ViscosityRegion(mask, count(mask))

#=

We therefore pass an [`InversionConfig`](@ref) and an [`InversionData`](@ref) to an [`InversionProblem`](@ref). Let's look at the initialized viscosity field:

Now that we have the displacement field, we can recover the viscosity field from which it results. We therefore pass an [`InversionConfig`](@ref) and an [`InversionData`](@ref) to an [`InversionProblem`](@ref). Let's look at the initialized viscosity field:
=#

# Defining prior with mean = 20.5, var = 0.5, bounds = (19.0, 22.0)
priors = combine_distributions([constrained_gaussian( "p_$(i)",
    20.5, 0.5, 19.0, 22.0) for i in 1:reduction.nparams])
paraminv = inversion_problem(deepcopy(fip), config, data, reduction, priors)
params = fill(10.0 ^ 20.5, count(mask))
reconstruct!(paraminv.fip, params, reduction)

function plot_viscfields(paraminv)
    cmap = cgrad(:jet, rev = true)
    crange = (19.5, 21.5)
    fig = Figure(size = (1800, 1000), fontsize = 40)
    axs = [Axis(fig[1,i], aspect = DataAspect()) for i in 1:2]
    [hidedecorations!(ax) for ax in axs]
    heatmap!(axs[1], log10.(true_viscosity), colormap = cmap, colorrange = crange)
    heatmap!(axs[2], log10.(paraminv.fip.p.effective_viscosity),
        colormap = cmap, colorrange = crange)
    Colorbar(fig[2, :], vertical = false, colormap = cmap, colorrange = crange,
        width = Relative(0.5))
    return fig
end
fig1 = plot_viscfields(paraminv)

#=
By performing a Kalman inversion, we can achieve a close match between ground truth and estimated viscosity field:
=#

solve!(paraminv)                # This will take a while
fig2 = plot_viscfields(paraminv)

#=
This remains an academic example, where we try to recover a known parameter field from data generated by the model itself. Nonetheless, the user should get a proof-of-concept and a scheme of how to implement such a procedure themself.
=#