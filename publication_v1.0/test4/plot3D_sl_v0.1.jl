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

# function main()
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

function get_bathitp()
    lon_bathy, lat_bathy, bathy = load_bathymetry()
    return linear_interpolation((lon_bathy, lat_bathy), bathy, extrapolation_bc = Flat())
end

bathitp = get_bathitp()

fig = Figure(resolution = (2300, 1400), fontsize = 40)
axs = [Axis(fig[i, j], aspect = DataAspect()) for j in 1:4, i in 2:3]
[hidedecorations!(ax) for ax in axs]
blugre = cgrad([:darkblue, :cornflowerblue, :skyblue1,
    :white, :aquamarine, :olivedrab, :darkgreen])
blupur = cgrad([:royalblue4, :royalblue1, :cyan,
    :white, :orchid1, :fuchsia, :purple])
G_opts = (colormap = blugre, colorrange = (-100, 100))
SL_opts = (colormap = blupur, colorrange = (-500, 500))
Colorbar(fig[1, 2:3], vertical = false, width = Relative(0.4); G_opts...)
Colorbar(fig[4, 2:3], vertical = false, width = Relative(0.4); SL_opts...)
for k in eachindex(tlaty)
    k_fi = argmin( (t .- tlaty[k]./1e3) .^ 2 )
    GSK1D = Gitp1D.(Lon, Lat, tlaty[k]) # - Gitp1D.(Lon, Lat, tlaty[1])
    GSK3D = Gitp3D.(Lon, Lat, tlaty[k]) # - Gitp3D.(Lon, Lat, tlaty[1])
    meansl1d = mean(SLitp1D.(Llon, Llat, tlaty[k]))
    meansl3d = mean(SLitp3D.(Llon, Llat, tlaty[k]))
    println("$(tlaty[k])  $meansl1d  $meansl3d")
    
    GFI1D = - fip1D.out.ue[k_fi] - fip1D.out.u[k_fi] + fip1D.out.geoid[k_fi] .+ meansl3d
    GFI3D = - fip.out.ue[k_fi] - fip.out.u[k_fi] + fip.out.geoid[k_fi] .+ meansl3d
    SLSK1D = SLitp1D.(Lon, Lat, tlaty[k])
    SLSK3D = SLitp3D.(Lon, Lat, tlaty[k])


    # GSK3D_max[k] = maximum(abs.(GSK3D))

    heatmap!(axs[1], GSK1D; G_opts...)
    heatmap!(axs[2], GSK3D; G_opts...)


    heatmap!(axs[5], SLSK1D; SL_opts...)
    heatmap!(axs[6], SLSK3D; SL_opts...)
    heatmap!(axs[7], GFI1D; SL_opts...)
    heatmap!(axs[8], GFI3D; SL_opts...)
    save("plots/test4/geoid/t=$(tlaty[k]).png", fig)

    # append!(ufi3D_vec, vec(ufastiso[mask]))
    # append!(ufi1D_vec, vec(ufastiso1D[mask]))
    # append!(usk_vec, vec(uskitp[mask]))
    # append!(usk1D_vec, vec(uskitp1D[mask]))

    # e3D = abs.(uskitp - ufastiso)
    # e3D_vec[k] .= vec(e3D[mask])
    # e1D = abs.(uskitp - ufastiso1D)
    # e1D_vec[k] .= vec(e1D[mask])
    # elaty1D = abs.(uskitp - uskitp1D)
    # elaty1D_vec[k] = vec(elaty1D[mask])

    # mean_error[k] = mean(e3D)
    # max_error[k] = maximum(e3D)
    # mean_error1D[k] = mean(e1D)
    # max_error1D[k] = maximum(e1D)
end

tlaty_vec = vcat([fill(k, nm) for k in eachindex(tlaty)]...)
e1D_mat = vcat(e1D_vec...)
e3D_mat = vcat(e3D_vec...)
elaty1D_mat = vcat(elaty1D_vec...)

fig = Figure(resolution = (2150, 2300), fontsize = 40)
ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
axbottom = [Axis(fig[i, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9],
    i in [4:6, 7:9]]
axs = vcat(ax_uu, ax_et, vec(axbottom))
umax = maximum(usk_max)
sk1D_color = :gray70

msmax = 4
ms = msmax .* (abs.(usk_vec) ./ umax) .^ 1.5 .+ 0.4
# rand_idx = rand(length(usk_vec)) .< pointspercentage
scatter!(axs[1], usk_vec, ufi1D_vec,
    markersize = ms, label = L"FI1D $\,$")
scatter!(axs[1], usk_vec, usk1D_vec,
    markersize = ms, label = L"SK1D $\,$", color = sk1D_color)
scatter!(axs[1], usk_vec, ufi3D_vec,
    markersize = ms, label = L"FI3D $\,$")
lines!(axs[1], -600:50, -600:50, color = :gray10, label = label = L"identity $\,$")
axs[1].xlabel = L"$u_\mathrm{SK3D}$ (m)"
axs[1].ylabel = L"$u_\mathrm{m}$ (m)"
axs[1].xticks = latexticks(-600:100:100)
axs[1].yticks = latexticks(-600:100:100)
xlims!(axs[1], (-600, 70))
ylims!(axs[1], (-600, 70))
axislegend(axs[1], position = :rb)
axs[1].title = L"(a) $\,$"

bgap = 0.25
widthfactor = 1.2
boxplot!(axs[2], tlaty_vec .- bgap, e1D_mat ./ umax, width = widthfactor*bgap,
    label = L"FI1D $\,$")
boxplot!(axs[2], tlaty_vec .+ bgap, elaty1D_mat ./ umax, width = widthfactor*bgap,
    label = L"SK1D $\,$", color = sk1D_color)
boxplot!(axs[2], tlaty_vec, e3D_mat ./ umax, width = widthfactor*bgap,
    label = L"FI3D $\,$")
axs[2].xlabel = L"Time (kyr) $\,$"
axs[2].ylabel = L"$e$ (1)"
axs[2].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
axs[2].yticks = latexticks(0:0.1:0.5)
axislegend(axs[2], position = :lt)
erellims = (0, 0.3)
ylims!(axs[2], erellims)
xlims!(axs[2], extrema(eachindex(tlaty)) .+ (-0.5, 0.5))
axs[2].yaxisposition = :right
axs[2].title = L"(b) $\,$"

ulims = (-500, 500)
elims = (-100, 100)
u_opts = (colorrange = ulims, colormap = :PuOr)
e_opts = (colorrange = elims, colormap = :lighttemperaturemap)
k = argmax(max_error1D)
k_fi = argmin( (t .- tlaty[k]/1e3) .^ 2 )
uskitp = itp.(Lon, Lat, tlaty[k])
ufastiso = fip1D.out.u[k_fi] + fip1D.out.ue[k_fi]
heatmap!(axs[3], uskitp; u_opts...)
heatmap!(axs[4], ufastiso; u_opts...)
heatmap!(axs[5], uskitp - ufastiso; e_opts...)
contour!(axs[3], mask; levels = [0.5], color = :gray10)
contour!(axs[4], mask; levels = [0.5], color = :gray10)
contour!(axs[5], mask; levels = [0.5], color = :gray10)
axs[3].title = L"(c) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[4].title = L"(d) FI1D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[5].title = L"(e) (SK3D - FI1D), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

k = argmax(max_error)
k_fi = argmin( (t .- tlaty[k]/1e3) .^ 2 )
uskitp = itp.(Lon, Lat, tlaty[k])
ufastiso = fip.out.u[k_fi] + fip.out.ue[k_fi]
hmu = heatmap!(axs[6], uskitp; u_opts...)
heatmap!(axs[7], ufastiso; u_opts...)
hme = heatmap!(axs[8], uskitp - ufastiso; e_opts...)
contour!(axs[6], mask; levels = [0.5], color = :gray10)
contour!(axs[7], mask; levels = [0.5], color = :gray10)
contour!(axs[8], mask; levels = [0.5], color = :gray10)
axs[6].title = L"(f) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[7].title = L"(g) FI3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[8].title = L"(h) (SK3D - FI3D), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

[hidedecorations!(ax) for ax in axbottom]
Colorbar(fig[10, 2:5], hmu, label = L"Vertical displacement (m) $\,$", vertical = false,
    flipaxis = false, width = Relative(0.6), ticks = latexticks(-500:250:500))
Colorbar(fig[10, 7:9], hme, label = L"$u_\mathrm{SK} - u_\mathrm{FI} $ (m)",
    vertical = false, flipaxis = false, width = Relative(0.8), ticks = latexticks(-100:50:100))
save("plots/test4/$case-N=$N-mask=$masktype-final_v0.4.png", fig)
save("plots/test4/$case-N=$N-mask=$masktype-final_v0.4.pdf", fig)
return nothing
# end

case = "3D-interactivesl=false-maxdepth=300000.0-nlayers=3-ICE6G_D"
N = 350
main(case, 350)

# cases = [
#     # "3D-interactivesl=false-maxdepth=300000.0-nlayers=3-ICE6G_D",
#     # "3D-interactivesl=false-maxdepth=400000.0-nlayers=3-ICE6G_D",
#     # "3D-interactivesl=false-maxdepth=500000.0-nlayers=3-ICE6G_D",
#     # "3D-interactivesl=false-maxdepth=600000.0-nlayers=3-ICE6G_D",
#     "3D-interactivesl=false-maxdepth=300000.0-nlayers=3-ICE6G_D"]
# for case in cases
#     main(case, 350)
# end
