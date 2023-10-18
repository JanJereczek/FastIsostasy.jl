push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("../helpers.jl")

n = 8
kernel = "cpu"
Omega = ComputationDomain(3000e3, n)
N = Omega.Nx
suffix = "viscmap_N$N"
dims, eta, eta_itp = load_wiens2021(Omega)
dimsT, _, Tlihto_itp = load_litho_thickness_laty()
Tlitho = Tlihto_itp.(Omega.Lon, Omega.Lat)


labels = [L"$\textbf{(%$char)}$" for char in ["a", "b", "c", "d"]]

xticks = (-3e6:1e6:3e6, latexify(-3:3))
yticks = (-3e6:1e6:3e6, latexify(-3:3))
visclim = (18, 23)
viscmap = cgrad(:jet, rev = true)
viscticks = (visclim[1]:visclim[2], latexify(visclim[1]:visclim[2]))

Xticks = [xticks, xticks, xticks, xticks, xticks]
Yticks = [yticks, yticks, yticks, yticks, yticks]
Xticksvisible = [true, true, true, true]
Yticksvisible = [true, false, false, true]
xlabels = [
    L"$x \: (10^3 \, \mathrm{km})$",
    L"$x \: (10^3 \, \mathrm{km})$",
    L"$x \: (10^3 \, \mathrm{km})$",
    L"$x \: (10^3 \, \mathrm{km})$",
]
ylabels = [
    L"$y \: (10^3 \, \mathrm{km})$",
    "",
    "",
    L"$y \: (10^3 \, \mathrm{km})$",
]
Xposition = [:bottom, :bottom, :bottom, :bottom]
Yposition = [:left, :left, :left, :right]




fig = Figure(resolution = (1350, 500), fontsize = 20)
nrows, ncols = 1, 4
axs = [Axis(
    fig[i, j],
    title = labels[(i-1)*ncols + j],
    titlegap = 10.0,
    xlabel = xlabels[(i-1)*ncols + j],
    ylabel = ylabels[(i-1)*ncols + j],
    xticks = Xticks[(i-1)*ncols + j],
    yticks = Yticks[(i-1)*ncols + j],
    xticklabelsvisible = Xticksvisible[(i-1)*ncols + j],
    yticklabelsvisible = Yticksvisible[(i-1)*ncols + j],
    yticksvisible = Yticksvisible[(i-1)*ncols + j],
    xaxisposition = Xposition[(i-1)*ncols + j],
    yaxisposition = Yposition[(i-1)*ncols + j],
    aspect = AxisAspect(1),
) for j in 1:ncols, i in 1:nrows]

for k in 1:3
    heatmap!(
        axs[k],
        Omega.X,
        Omega.Y,
        eta[:, :, k],
        colormap = viscmap,
        colorrange = visclim,
    )
    heatmap!(
        axs[k],
        Omega.X,
        Omega.Y,
        Tlitho .> dims[3][k] / 1e3,
        colormap = cgrad([:gray80, :gray80]),
        colorrange = (0.9, 1.1),
        lowclip = :transparent,
        highclip = :transparent,
    )
end

hm = heatmap!(
    axs[4],
    Omega.X,
    Omega.Y,
    Tlitho,
    colorrange = (50, 250),
)

Colorbar(
    fig[2, 1:3],
    colormap = viscmap,
    colorrange = visclim,
    vertical = false,
    width = Relative(0.3),
    label = L"Log$_{10}$-viscosity ($\mathrm{Pa \, s})$",
    ticks = viscticks,
    flipaxis = false,
)

Colorbar(
    fig[2, 4],
    hm,
    vertical = false,
    width = Relative(0.9),
    label = L"Lithospheric thickness (km) $\,$",
    ticks = latexticks(50:50:250),
    flipaxis = false,
)
colgap!(fig.layout, 10)
save("plots/viscmaps/$suffix.pdf", fig)