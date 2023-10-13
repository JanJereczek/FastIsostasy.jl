using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("laty.jl")
include("../helpers.jl")

function load_1D_results(N)
    @load "../data/test4/ICE6G/1D-interactivesl=false-ICE6G_D"*
        "-N=$N.jld2" t fip Hitp Hice_vec deltaH
    return t, fip
end

function load_3D_results(case, N)
    @load "../data/test4/ICE6G/$case-N=$N.jld2" t fip Hitp Hice_vec deltaH
    return t, fip, Hitp, Hice_vec, deltaH
end

case = "3D-interactivesl=true-maxdepth=300000.0-nlayers=3-ICE6G_D"
N = 64
t, fip, Hitp, Hice_vec, deltaH = load_3D_results(case, N)
t1D, fip1D = load_1D_results(N);
tlaty, _, _, _, Ritp3D = load_laty_ICE6G(case = "3D", variable = "R")
_, _, _, _, Ritp1D = load_laty_ICE6G(case = "1D", variable = "R")
tlaty, _, _, _, SLitp3D = load_laty_ICE6G(case = "3D", variable = "SL")
_, _, _, _, SLitp1D = load_laty_ICE6G(case = "1D", variable = "SL")

Omega, p = reinit_structs_cpu(fip.Omega, fip.p)
Lon, Lat = Omega.Lon, Omega.Lat
Llon, Llat = meshgrid(-180.0:180.0, -70.0:-50.0)
X, Y = Omega.X, Omega.Y
tice6g, _, _, _, Hice_itp = load_ice6gd()

fig = Figure(resolution = (2300, 1400), fontsize = 40)
axs = [Axis(fig[i, j], aspect = DataAspect()) for j in 1:4, i in 2:3]
[hidedecorations!(ax) for ax in axs]
blugre = cgrad([:darkblue, :cornflowerblue, :skyblue1,
    :white, :aquamarine, :olivedrab, :darkgreen], 17, categorical = true)
blupur = cgrad([:royalblue4, :royalblue1, :cyan,
    :white, :orchid1, :fuchsia, :purple], 17, categorical = true)
G_opts = (colormap = blugre, colorrange = (-100, 100))
SL_opts = (colormap = blupur, colorrange = (-200, 200))
Colorbar(fig[1, 2:3], vertical = false, width = Relative(0.4); G_opts...)
Colorbar(fig[4, 2:3], vertical = false, width = Relative(0.4); SL_opts...)
for k in eachindex(tlaty)
    k_fi = argmin( (t .- tlaty[k]./1e3) .^ 2 )
    meansl3d = mean(SLitp3D.(Llon, Llat, tlaty[k]))
    SLSK1D = SLitp1D.(Lon, Lat, tlaty[k]) + Ritp1D.(Lon, Lat, tlaty[k]) .- meansl3d
    SLSK3D = SLitp3D.(Lon, Lat, tlaty[k]) + Ritp3D.(Lon, Lat, tlaty[k]) .- meansl3d
    
    SLFI1D = fip1D.out.geoid[k_fi]
    SLFI3D = fip.out.geoid[k_fi]

    heatmap!(axs[5], SLSK1D; SL_opts...)
    heatmap!(axs[6], SLSK3D; SL_opts...)
    heatmap!(axs[7], SLFI1D; SL_opts...)
    heatmap!(axs[8], SLFI3D; SL_opts...)
    save("plots/test4/sl_relative/t=$(tlaty[k]).png", fig)
end