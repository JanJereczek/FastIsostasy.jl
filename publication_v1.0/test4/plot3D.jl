using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("laty.jl")
include("../helpers.jl")

case = "3D"
N = 64
@load "../data/test4/ICE6G/$case-interactivesl=true-N=$N.jld2" t fip Hitp Hice_vec deltaH
make_anims = false

Hlim = (1e-8, 4e3)
kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
axs = [Axis(fig[1, j], title = @lift("t = $(t[$kobs]) ka"),
    aspect = DataAspect()) for j in 1:2]
[hidedecorations!(ax) for ax in axs]
hm1 = heatmap!(axs[1], @lift(Hice_vec[$kobs]), colorrange = Hlim, lowclip = :transparent,
    colormap = :ice)
hm2 = heatmap!(axs[2], @lift(fip.out.u[$kobs]), colormap = :vik, colorrange = (-800, 800))
Colorbar(fig[2, 1], hm1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], hm2, vertical = false, flipaxis = false, width = Relative(0.8))
if make_anims
    record(fig, "plots/test4/$case-N=$N-ICE6G-cycle-displacement.mp4",
        eachindex(t), framerate = 10) do k
        kobs[] = k
    end
end

dHlims = (-2600, 2600)
ulims = (-500, 500)
kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
vars = [deltaH, fip.out.u .+ fip.out.ue]
axs = [Axis3(fig[1, j], title = @lift("t = $(t[$kobs]) ka,"*
 "  range = $(round.(extrema(vars[j][$kobs])))")) for j in 1:2]
# [hidedecorations!(ax) for ax in axs]
sf1 = surface!(axs[1], @lift(deltaH[$kobs]), colorrange = dHlims, colormap = :vik)
sf2 = surface!(axs[2], @lift(fip.out.u[$kobs] + fip.out.ue[$kobs]), colormap = :PuOr,
    colorrange = ulims)
zlims!(axs[1], dHlims)
zlims!(axs[2], ulims)
Colorbar(fig[2, 1], sf1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], sf2, vertical = false, flipaxis = false, width = Relative(0.8))
if make_anims
    record(fig, "plots/test4/$case-N=$N-ICE6G-cycle-surface.mp4",
        eachindex(t), framerate = 10) do k
        kobs[] = k
    end
end

tlaty, ulaty, Lon, Lat, itp = load_laty_ICE6G(case = case)

elims = (-100, 100)
mean_error = fill(Inf, length(tlaty))
max_error = fill(Inf, length(tlaty))
ulaty_max = fill(Inf, length(tlaty))
for k in eachindex(tlaty)
    k_fastiso = argmin( (t .- tlaty[k]./1e3) .^ 2 )
    ulatyitp = itp.(fip.Omega.Lon, fip.Omega.Lat, tlaty[k])
    ulaty_max[k] = maximum(abs.(ulatyitp))
    ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
    mean_error[k] = mean( abs.(ulatyitp - ufastiso) )
    max_error[k] = maximum( abs.(ulatyitp - ufastiso) )
    tmpfig = Figure(resolution = (1800, 700), fontsize = 30)
    axs = [Axis(tmpfig[1, j]) for j in 1:3]
    [hidedecorations!(ax) for ax in axs]
    hm1 = heatmap!(axs[1], ulatyitp, colorrange = ulims, colormap = :PuOr)
    hm2 = heatmap!(axs[2], ufastiso, colorrange = ulims, colormap = :PuOr)
    hm3 = heatmap!(axs[3], ulatyitp - ufastiso, colorrange = elims,
        colormap = :lighttemperaturemap)
    Colorbar(tmpfig[2, 1:2], hm1, label = "vertical displacement (m)", vertical = false,
        width = Relative(0.4))
    Colorbar(tmpfig[2, 3], hm3, label = L"$u_\mathrm{sk} - u_\mathrm{fi} $ (m)",
        vertical = false, width = Relative(0.8))
    save("plots/test4/$case-N=$N-laty-ICE6G-local$(tlaty[k]).png", tmpfig)
end


fig = Figure(resolution = (2150, 1200), fontsize = 34)
axtop = Axis(fig[1:2, 2:5], xticklabelrotation = π/2)
# axfill = [Axis(fig[1:2, 1]), Axis(fig[1:2, 6])]
# hidedecorations!.(axfill)
# hidespines!.(axfill)
axbottom = [Axis(fig[3:6, j], aspect = DataAspect()) for j in [1:2, 3:4, 5:6]]
axs = vcat(axtop, axbottom)
barplot!(axs[1], eachindex(tlaty), max_error ./ maximum(ulaty_max), label = L"max $\,$")
barplot!(axs[1], eachindex(tlaty), mean_error ./ maximum(ulaty_max), label = L"mean $\,$")
axs[1].xlabel = L"Time (kyr) $\,$"
axs[1].ylabel = L"Rel. difference (m) $\,$"
axs[1].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
axs[1].yticks = latexticks(0:0.1:0.5)
axislegend(axs[1], position = :lt)

erellims = (0, 0.3)
ylims!(axs[1], erellims)

k = argmax(max_error)
k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
ulatyitp = itp.(fip.Omega.Lon, fip.Omega.Lat, tlaty[k])
ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
[hidedecorations!(ax) for ax in axbottom]
hm1 = heatmap!(axs[2], ulatyitp, colorrange = ulims, colormap = :PuOr)
hm2 = heatmap!(axs[3], ufastiso, colorrange = ulims, colormap = :PuOr)
hm3 = heatmap!(axs[4], ulatyitp - ufastiso, colorrange = (-100, 100),
    colormap = :lighttemperaturemap)
Colorbar(fig[7, 2:3], hm1, label = L"Vertical displacement (m) $\,$", vertical = false,
    flipaxis = false, width = Relative(0.8), ticks = latexticks(-500:250:500))
Colorbar(fig[7, 5:6], hm3, label = L"$u_\mathrm{sk} - u_\mathrm{fi} $ (m)",
    vertical = false, flipaxis = false, width = Relative(0.8), ticks = latexticks(-100:50:100))
axs[2].title = L"Seakon at $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[3].title = L"FastIsostasy at $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[4].title = L"Difference at $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
save("plots/test4/$case-N=$N-final.png", fig)