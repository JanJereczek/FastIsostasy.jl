push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using CairoMakie
using Interpolations
include("external_viscosity_maps.jl")
include("helpers_plot.jl")

n = 6
use_cuda = false
case = "scaledviscosity"

T = Float64
L = T(3000e3)                       # half-length of the square domain (m)
Omega = ComputationDomain(L, n)     # domain parameters
c = PhysicalConstants()
if occursin("homogeneous", case)
    channel_viscosity = fill(1e20, Omega.N, Omega.N)
    halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
    lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
    p = MultilayerEarth(
        Omega,
        c,
        layers_viscosity = lv,
    )
elseif occursin("meanviscosity", case)
    log10_eta_channel = interpolate_visc_wiens_on_grid(Omega.X, Omega.Y)
    channel_viscosity = 10 .^ (log10_eta_channel)
    halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
    lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
    p = MultilayerEarth(
        Omega,
        c,
        layers_viscosity = lv,
    )
elseif occursin("scaledviscosity", case)
    lb = [88e3, 180e3, 280e3, 400e3]
    halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)

    Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        # [eta_interpolators[1].(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    p = MultilayerEarth(
        Omega,
        c,
        layers_begin = lb,
        layers_viscosity = lv,
    )
end

lv = [p.layers_viscosity[:, :, i] for i in axes(p.layers_viscosity, 3)[3:end]]
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