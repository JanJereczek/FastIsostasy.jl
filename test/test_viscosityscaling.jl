using FastIsostasy
using JLD2
using CairoMakie
using Interpolations
include("helpers/viscmaps.jl")
include("helpers/plot.jl")

n = 6
use_cuda = false
case = "scaledviscosity"

T = Float64
W = T(3000e3)                       # half-length of the square domain (m)
Omega = ComputationDomain(W, n)     # domain parameters
c = PhysicalConstants()
if occursin("homogeneous", case)
    channel_viscosity = fill(1e20, Omega.Nx, Omega.Ny)
    halfspace_viscosity = fill(1e21, Omega.Nx, Omega.Ny)
    lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
    p = LayeredEarth(Omega, layer_viscosities = lv)
elseif occursin("meanviscosity", case)
    log10_eta_channel = interpolate_visc_wiens_on_grid(Omega.X, Omega.Y)
    channel_viscosity = 10 .^ (log10_eta_channel)
    halfspace_viscosity = fill(1e21, Omega.Nx, Omega.Ny)
    lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
    p = LayeredEarth(Omega, layer_viscosities = lv)
elseif occursin("scaledviscosity", case)
    lb = [88e3, 180e3, 280e3, 400e3]
    halfspace_logviscosity = fill(21.0, Omega.Nx, Omega.Ny)

    Eta, Eta_mean, z = load_wiens2022()
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        # [eta_interpolators[1].(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
end

lv = [p.layer_viscosities[:, :, i] for i in axes(p.layer_viscosities, 3)[3:end]]
push!(lv, p.effective_viscosity)

fig = Figure(resolution = (2000, 500), fontsize = 20)
axs = [Axis(
    fig[1, j],
    aspect = AxisAspect(1),
) for j in eachindex(lv)]
visclim = (18, 23)
viscmap = cgrad(:jet)
viscticks = (visclim[1]:visclim[2], num2latexstring.(visclim[1]:visclim[2]))

for k in eachindex(lv)
    heatmap!(
        axs[k],
        Omega.X,
        Omega.Y,
        log10.(lv[k])',
        colormap = viscmap,
        colorrange = visclim,
    )
end

Colorbar(
    fig[2, :],
    colormap = viscmap,
    colorrange = visclim,
    vertical = false,
    width = Relative(0.3),
    label = L"Log$_{10}$-viscosity ($\mathrm{Pa \, s})$",
    ticks = viscticks,
)