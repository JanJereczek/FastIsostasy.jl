using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

@load "../data/test4/ICE6G/1D-interactivesl=true-bsl=external-N=350.jld2" t fip Hitp Hice_vec
(lonlaty, latlaty, tlaty), ulaty, itp = load_latychev2023_ICE6G()
tlaty = tlaty[vcat(1:11, [13, 15, 16, 17, 18, 20, 22, 24])]
Omega = ComputationDomain(3500e3, 3500e3, 350, 350, use_cuda=false)
Lon, Lat = Omega.Lon, Omega.Lat
Hice = [Hitp.(Lon, Lat, tk/1e3) for tk in tlaty]
kmax = argmax([mean(Hice[k]) for k in eachindex(Hice)])
mask = (Hice[kmax] .> 1) .|| (Omega.R .< 500e3)

elims = (-100, 100)
mean_error = fill(Inf, length(tlaty))
max_error = fill(Inf, length(tlaty))
ufi1D_vec = Float64[]
usk1D_vec = Float64[]

for k in eachindex(tlaty)
    k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
    ulatyitp = itp.(Lon, Lat, tlaty[k])
    ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]

    mean_error[k] = mean( abs.(ulatyitp - ufastiso) )
    max_error[k] = maximum( abs.(ulatyitp - ufastiso) )

    append!(ufi1D_vec, vec(ufastiso[mask]))
    append!(usk1D_vec, vec(ulatyitp[mask]))
end

fig = Figure(size = (2150, 1500), fontsize = 40)

ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
axbottom = [Axis(fig[4:6, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9]]
[hidedecorations!(ax) for ax in axbottom]
axs = vcat(ax_uu, ax_et, vec(axbottom))
umax = maximum(abs.(usk1D_vec))

msmax = 4
ms = msmax .* (abs.(usk1D_vec) ./ umax) .^ 1.5 .+ 0.4
scatter!(axs[1], usk1D_vec, ufi1D_vec,
    markersize = ms, label = L"FI1D $\,$")
lines!(axs[1], -500:50, -500:50, color = :gray10, label = label = L"identity $\,$")
axs[1].xlabel = L"$u_\mathrm{SK1D}$ (m)"
axs[1].ylabel = L"$u_\mathrm{FI1D}$ (m)"
axs[1].xticks = latexticks(-600:100:100)
axs[1].yticks = latexticks(-600:100:100)
xlims!(axs[1], (-450, 20))
ylims!(axs[1], (-450, 20))
# axislegend(axs[1], position = :rb)
axs[1].title = L"(a) $\,$"

barplot!(axs[2], eachindex(tlaty), max_error ./ umax, label = L"max $\,$", color = :dodgerblue1)
barplot!(axs[2], eachindex(tlaty), mean_error ./ umax, label = L"mean $\,$", color = :dodgerblue3)
axs[2].title = L"(b) $\,$"
axs[2].xlabel = L"Time (kyr) $\,$"
axs[2].ylabel = L"$e$ (1)"
axs[2].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
axs[2].yticks = latexticks(0:0.1:0.5)
axs[2].yaxisposition = :right
axislegend(axs[2], position = :rt)
erellims = (0, 0.3)
ylims!(axs[2], erellims)
xlims!(axs[2], extrema(eachindex(tlaty)) .+ (-0.5, 0.5))


ulims = (-500, 500)
elims = (-100, 100)
u_opts = (colorrange = ulims, colormap = :PuOr)
e_opts = (colorrange = elims, colormap = :lighttemperaturemap)
k = argmax(max_error)
k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
ulatyitp = itp.(Lon, Lat, tlaty[k])
ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
heatmap!(axs[3], ufastiso; u_opts...)
heatmap!(axs[4], ulatyitp; u_opts...)
heatmap!(axs[5], ulatyitp - ufastiso; e_opts...)
contour!(axs[3], mask; levels = [0.5], color = :gray10)
contour!(axs[4], mask; levels = [0.5], color = :gray10)
contour!(axs[5], mask; levels = [0.5], color = :gray10)
axs[3].title = L"(d) FI1D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[4].title = L"(c) SK1D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[5].title = L"(e) (SK1D - FI1D), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
save("plots/test4/1D_v0.4.png", fig)