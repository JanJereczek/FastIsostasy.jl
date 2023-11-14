using FastIsostasy, JLD2, CairoMakie

function load_results(maxdepth, nlayers, N, interactive_sl::Bool)
    @load "../data/test4/ICE6G/3D-interactivesl=$interactive_sl-maxdepth=$maxdepth"*
            "-nlayers=$nlayers-N=$(N)-premparams.jld2" fip
    return fip
end

maxdepth, nlayers, N = 500e3, 3, 64
fip0 = load_results(maxdepth, nlayers, N, false)
fip1 = load_results(maxdepth, nlayers, N, true)

fig = Figure(resolution = (1800, 800))
axs = [Axis(fig[1, j], aspect = DataAspect()) for j in 1:3]
[hidedecorations!(ax) for ax in axs]
uopts = (colormap = :PuOr, colorrange = (-400, 400))
eopts = (colormap = :lighttemperaturemap, colorrange = (-50, 50))

tplot = -14e3
k = argmin( (seconds2years.(fip1.out.t) .- tplot) .^ 2 )
u0 = fip0.out.u[k] + fip0.out.ue[k]
u1 = fip1.out.u[k] + fip1.out.ue[k]
heatmap!(axs[1], u0; uopts...)
heatmap!(axs[2], u1; uopts...)
heatmap!(axs[3], u1 - u0; eopts...)
Colorbar(fig[2, 1:2], vertical = false, width = Relative(0.4); uopts...)
Colorbar(fig[2, 3], vertical = false, width = Relative(0.8); eopts...)
fig
save("../plots/test4/comparesl-N=$N-maxdepth=$maxdepth-nlayers=$nlayers")




tplot = -30e3
k = argmin( (seconds2years.(fip1.out.t) .- tplot) .^ 2 )
fig = Figure(resolution = (1800, 800))
axs = [Axis(fig[1, j], aspect = DataAspect()) for j in 1:3]
[hidedecorations!(ax) for ax in axs]
uopts = (colormap = :PuOr, colorrange = (-50, 50))
eopts = (colormap = :lighttemperaturemap, colorrange = (-50, 50))

heatmap!(axs[1], fip0.out.seasurfaceheight[k]; uopts...)
heatmap!(axs[2], fip1.out.seasurfaceheight[k] .- mean(fip1.out.seasurfaceheight[k]); uopts...)
heatmap!(axs[3], fip1.out.seasurfaceheight[k] - fip0.out.seasurfaceheight[k]; eopts...)
Colorbar(fig[2, 1:2], vertical = false, width = Relative(0.4); uopts...)
Colorbar(fig[2, 3], vertical = false, width = Relative(0.8); eopts...)
fig