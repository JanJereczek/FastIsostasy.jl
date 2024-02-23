using FastIsostasy
using CairoMakie, JLD2, NCDatasets, Interpolations
include("../helpers.jl")

path_bedmachine = "../data/BedMachine/ANT-16KM_TOPO-BedMachine.nc"
ds = NCDataset(path_bedmachine)
xb = Float64.(copy(ds["xc"][:]))
yb = Float64.(copy(ds["yc"][:]))
z_srf = Float64.(copy(ds["z_srf"][:,:]))
mask = Float64.(copy(ds["mask"][:,:]))
close(ds)

mask_itp = linear_interpolation((xb, yb), mask)
z_srf_itp = linear_interpolation((xb, yb), z_srf)

n = 8
kernel = "cpu"
Omega = ComputationDomain(3000e3, n)
c = PhysicalConstants()
N = Omega.Nx
suffix = "viscmap_N$N"
lb = collect(100e3:100e3:300e3)

viscdata = "Wiens2022"

if viscdata == "Wiens2022"
    dims, logeta, logeta_itp = load_wiens2022()
    logeta = cat([logeta_itp.(Omega.X, Omega.Y, z) for z in lb]..., dims=3)
elseif viscdata == "Pan2022"
    dims, logeta, logeta_itp = load_logvisc_pan2022()
    zz = c.r_equator .- lb
    logeta = cat([logeta_itp.(Omega.Lon, Omega.Lat, z) for z in zz]..., dims=3)
end

dimsT, _, Tlihto_itp = load_lithothickness_pan2022()
Tlitho = Tlihto_itp.(Omega.Lon, Omega.Lat)
antmask = mask_itp.(Omega.X ./ 1e3, Omega.Y ./ 1e3)
zsrf = z_srf_itp.(Omega.X ./ 1e3, Omega.Y ./ 1e3)

labels = [L"(%$char) $\,$" for char in ["a", "b", "c", "d"]]

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

fig = Figure(size = (1350, 530), fontsize = 24)
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
        logeta[:, :, k],
        colormap = viscmap,
        colorrange = visclim,
    )
    heatmap!(
        axs[k],
        Omega.X,
        Omega.Y,
        Tlitho .> lb[k] / 1e3,
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


for k in 1:4
    contour!(
        axs[k],
        Omega.X[:, 1],
        Omega.Y[1, :],
        zsrf,
        levels = [200],
        color = [:gray50],
        linewidth = 3,
    )
    contour!(
        axs[k],
        Omega.X[:, 1],
        Omega.Y[1, :],
        zsrf,
        levels = [20],
        color = [:black],
        linewidth = 3,
    )
    # contour!(
    #     axs[k],
    #     Omega.X[:, 1],
    #     Omega.Y[1, :],
    #     antmask .<= 1,
    #     levels = [0.99],
    #     color = [:black],
    #     linewidth = 5,
    # )
end

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
colgap!(fig.layout, 5)
rowgap!(fig.layout, 1, 15)
save("plots/viscmaps/$suffix-v0.3.pdf", fig)