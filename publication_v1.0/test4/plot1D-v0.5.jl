using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers_plot.jl")

function load_elva_displacement(N)
    ds = NCDataset("../data/test4/ICE6G/1D-interactivesl=true-maskbsl=true-N=$N.nc", "r")
    t = ds["t"][:]
    u = ds["u"][:, :, :] + ds["ue"][:, :, :]
    mask = ds["active mask"][:, :]
    Hice = ds["Hice"][:, :, :]
    return t, u, Bool.(mask), Hice
end

function load_elra_displacement(N)
    ds = NCDataset("../data/test4/ICE6G/elra-interactivesl=true-maskbsl=true-N=$N.nc", "r")
    u = ds["u"][:, :, :] + ds["ue"][:, :, :]
    close(ds)
    return u
end

N = 350
t, u, mask, Hice = load_elva_displacement(N)
u_elra = load_elra_displacement(N)

(lonlaty, latlaty, tlaty), ulaty, itp = load_latychev2023_ICE6G()
tlaty = Int.(tlaty[vcat(1:11, [13, 15, 16, 17, 18, 20, 22, 24])])
Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = false)
Lon, Lat = Omega.Lon, Omega.Lat
klgm = argmax([mean(H) for H in eachslice(Hice, dims = 3)])
lgm = ((Hice[:, :, klgm] .> 1) .|| (Omega.R .< 500e3)) .&& (Omega.Y .> -2_400e3)

elims = (-100, 100)
elva_mean_error = fill(Inf, length(tlaty))
elva_max_error = fill(Inf, length(tlaty))
elra_mean_error = fill(Inf, length(tlaty))
elra_max_error = fill(Inf, length(tlaty))

uelva_vec = Float64[]
uelra_vec = Float64[]
usk1D_vec = Float64[]

for k in eachindex(tlaty)
    k_fastiso = argmin( (t .- tlaty[k]) .^ 2 )
    ulatyitp = itp.(Lon, Lat, tlaty[k])

    elva_mean_error[k] = mean( abs.(ulatyitp - u[:, :, k_fastiso]) )
    elva_max_error[k] = maximum( abs.(ulatyitp - u[:, :, k_fastiso]) )
    elra_mean_error[k] = mean( abs.(ulatyitp - u_elra[:, :, k_fastiso]) )
    elra_max_error[k] = maximum( abs.(ulatyitp - u_elra[:, :, k_fastiso]) )

    append!(uelva_vec, vec(u[:, :, k_fastiso][mask]))
    append!(uelra_vec, vec(u_elra[:, :, k_fastiso][mask]))
    append!(usk1D_vec, vec(ulatyitp[mask]))
end

fig = Figure(size = (2100, 2300), fontsize = 40)
elra_color = :red
elva_color = :dodgerblue3

ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
ax_elra = [Axis(fig[4:6, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9]]
ax_elva = [Axis(fig[7:9, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9]]
[hidedecorations!(ax) for ax in ax_elra]
[hidedecorations!(ax) for ax in ax_elva]
axs = vcat(ax_uu, ax_et, vec(ax_elra), vec(ax_elva))
umax = maximum(abs.(usk1D_vec))

msmax = 4
ms = msmax .* (abs.(usk1D_vec) ./ umax) .^ 1.5 .+ 0.4
scatter!(axs[1], usk1D_vec, uelra_vec, markersize = ms, label = L"ELRA $\,$", color = elra_color)
scatter!(axs[1], usk1D_vec, uelva_vec, markersize = ms, label = L"ELVA $\,$")
lines!(axs[1], -500:50, -500:50, color = :gray10, label = label = L"identity $\,$")
axs[1].title = L"(a) $\,$"
axs[1].xlabel = L"$u_\mathrm{SK1D}$ (m)"
axs[1].ylabel = L"$u_m$ (m)"
axs[1].xticks = latexticks(-600:100:100)
axs[1].yticks = latexticks(-600:100:100)
xlims!(axs[1], (-500, 20))
ylims!(axs[1], (-500, 20))
axislegend(axs[1], position = :rb)

max_opts = (marker = :utriangle, linestyle = :solid, markersize = 30, linewidth = 5)
mean_opts = (marker = :circle, linestyle = :dash, markersize = 30, linewidth = 5)
scatterlines!(axs[2], eachindex(tlaty), elra_max_error ./ umax,
    label = L"$\hat{e}_\mathrm{ELRA}$", color = elra_color; max_opts...)
scatterlines!(axs[2], eachindex(tlaty), elra_mean_error ./ umax,
    label = L"$\bar{e}_\mathrm{ELRA}$", color = elra_color; mean_opts...)
scatterlines!(axs[2], eachindex(tlaty), elva_max_error ./ umax,
    label = L"$\hat{e}_\mathrm{ELVA}$", color = elva_color; max_opts...)
scatterlines!(axs[2], eachindex(tlaty), elva_mean_error ./ umax,
    label = L"$\bar{e}_\mathrm{ELVA}$", color = elva_color; mean_opts...)
axs[2].title = L"(b) $\,$"
axs[2].xlabel = L"Time (kyr) $\,$"
axs[2].ylabel = L"$e$ (1)"
axs[2].xticks = (eachindex(tlaty), latexify(Int.(round.(tlaty ./ 1e3, digits = 3))))
axs[2].yticks = latexticks(0:0.1:0.5)
axs[2].yaxisposition = :right
erellims = (0, 0.3)
ylims!(axs[2], erellims)
xlims!(axs[2], extrema(eachindex(tlaty)) .+ (-0.5, 0.5))
axislegend(axs[2], position = :lt, nbanks = 2)


ulims = (-500, 500)
elims = (-100, 100)
u_opts = (colorrange = ulims, colormap = :PuOr)
e_opts = (colorrange = elims, colormap = :lighttemperaturemap)

k = argmax(elra_max_error)
kmax = argmin( (t .- tlaty[k]) .^ 2 )
ulatyitp = itp.(Lon, Lat, tlaty[k])
heatmap!(axs[3], u_elra[:, :, kmax]; u_opts...)
hmu = heatmap!(axs[4], ulatyitp; u_opts...)
hme = heatmap!(axs[5], ulatyitp - u_elra[:, :, kmax]; e_opts...)
contour!(axs[3], lgm; levels = [0.5], color = :gray50, linewidth = 2)
contour!(axs[4], lgm; levels = [0.5], color = :gray50, linewidth = 2)
contour!(axs[5], lgm; levels = [0.5], color = :gray50, linewidth = 2)
contour!(axs[3], mask; levels = [0.5], color = :gray10, linewidth = 2)
contour!(axs[4], mask; levels = [0.5], color = :gray10, linewidth = 2)
contour!(axs[5], mask; levels = [0.5], color = :gray10, linewidth = 2)
axs[3].title = L"(c) ELRA, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[4].title = L"(d) SK1D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[5].title = L"(e) (SK1D - ELRA), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

k = argmax(elva_max_error)
kmax = argmin( (t .- tlaty[k]) .^ 2 )
ulatyitp = itp.(Lon, Lat, tlaty[k])
heatmap!(axs[6], u[:, :, kmax]; u_opts...)
hmu = heatmap!(axs[7], ulatyitp; u_opts...)
hme = heatmap!(axs[8], ulatyitp - u[:, :, kmax]; e_opts...)
contour!(axs[6], lgm; levels = [0.5], color = :gray50, linewidth = 2)
contour!(axs[7], lgm; levels = [0.5], color = :gray50, linewidth = 2)
contour!(axs[8], lgm; levels = [0.5], color = :gray50, linewidth = 2)
contour!(axs[6], mask; levels = [0.5], color = :gray10, linewidth = 2)
contour!(axs[7], mask; levels = [0.5], color = :gray10, linewidth = 2)
contour!(axs[8], mask; levels = [0.5], color = :gray10, linewidth = 2)
axs[6].title = L"(f) ELVA, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[7].title = L"(g) SK1D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[8].title = L"(h) (SK1D - ELVA), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

Colorbar(fig[10, 1:3], hmu, label = L"Vertical displacement (m) $\,$", vertical = false,
flipaxis = false, width = Relative(0.8), height = Relative(0.15),
ticks = latexticks(-500:250:500))
Colorbar(fig[10, 7:9], hme, label = L"Difference to SK3D (m) $\,$",
vertical = false, flipaxis = false, width = Relative(0.8),
height = Relative(0.15), ticks = latexticks(-100:50:100))

elem_0 = LineElement(color = :transparent, linewidth = 4)
elem_1 = LineElement(color = :gray10, linewidth = 4)
elem_2 = LineElement(color = :gray50, linewidth = 4)
Legend(fig[10, 4:6],
    [elem_0, elem_0, elem_1, elem_2],
    ["", "", L"Boundary of active mask $\,$", L"LGM extent of AIS $\,$"],
    patchsize = (35, 35), rowgap = 10, framevisible = false)
rowgap!(fig.layout, 9, -30.0)
colgap!(fig.layout, 10.0)

save("plots/test4/1D_v0.5.png", fig)