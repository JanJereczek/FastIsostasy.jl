using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("laty.jl")
include("compute5C.jl")

# @load "../data/test5/ICE6G/homogeneous-N=200.jld2" t fip Hitp Hice_vec deltaH
@load "../data/test5/ICE6G/homogeneous-interactivesl=true-N=128.jld2" t fip Hitp Hice_vec deltaH

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
record(fig, "plots/test5/ICE6G-cycle-displacement.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
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
record(fig, "plots/test5/ICE6G-cycle-surface.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
end

tlaty, ulaty, Lon, Lat, itp = load_laty_ICE6G()

mean_error = fill(Inf, length(tlaty))
max_error = fill(Inf, length(tlaty))
for k in eachindex(tlaty)
    k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
    ulatyitp = itp.(fip.Omega.Lon, fip.Omega.Lat, tlaty[k])
    ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
    mean_error[k] = mean( abs.(ulatyitp - ufastiso) )
    max_error[k] = maximum( abs.(ulatyitp - ufastiso) )
    tmpfig = Figure(resolution = (1800, 700), fontsize = 30)
    axs = [Axis(tmpfig[1, j]) for j in 1:3]
    [hidedecorations!(ax) for ax in axs]
    hm1 = heatmap!(axs[1], ulatyitp, colorrange = (-500, 500), colormap = :PuOr)
    hm2 = heatmap!(axs[2], ufastiso, colorrange = (-500, 500), colormap = :PuOr)
    hm3 = heatmap!(axs[3], ulatyitp - ufastiso, colorrange = (-100, 100),
        colormap = :lighttemperaturemap)
    Colorbar(tmpfig[2, 1:2], hm1, label = "vertical displacement (m)", vertical = false,
        width = Relative(0.4))
    Colorbar(tmpfig[2, 3], hm3, label = L"$u_\mathrm{sk} - u_\mathrm{fi} $ (m)",
        vertical = false, width = Relative(0.8))
    save("plots/test5/laty-ICE6G-local$(tlaty[k]).png", tmpfig)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "time (kyr)", ylabel = "absolute difference (m)")
lines!(ax, tlaty, mean_error, label = "mean")
lines!(ax, tlaty, max_error, label = "max")
axislegend(ax)
save("plots/test5/ICE6G-difftimeseries.png", fig)

barplot(eachindex(tlaty), max_error)
barplot(eachindex(tlaty), mean_error)