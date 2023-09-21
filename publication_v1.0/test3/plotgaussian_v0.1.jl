push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("../../test/helpers/plot.jl")
include("../helpers.jl")

function main(n::Int;  kernel = "cpu")

    N = 2^n
    suffix = "$(kernel)_Nx$(N)_Ny$(N)_dense"
    prefixes = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η"]
    nf = length(prefixes)

    rigidity_map = cgrad([:royalblue, :white, :firebrick1])
    viscosity_map = cgrad([:purple4, :white, :darkorange2])
    cmaps = [rigidity_map, rigidity_map, viscosity_map, viscosity_map]
    clims = [(50e3, 250e3), (50e3, 250e3), (20, 22), (20, 22)]
    clabels = [L"Lithospheric thickness $T_0$ (km)",
               L"Upper-mantle log-viscosity $\eta_\mathrm{eff}$ ($\mathrm{Pa \, s}$)"]

    xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))
    yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3))

    fig = Figure(resolution = (1200, 500), fontsize = 24)
    axs = [Axis(fig[1, j], aspect = DataAspect(),
        xticks = xticks, yticks = yticks) for j in 1:nf]
    for k in 1:nf
        @load "../data/test3/$(prefixes[k])_$suffix.jld2" fip
        if k <= 2
            plotparam = fip.p.litho_thickness
        elseif k <= 4
            plotparam = log10.(fip.p.effective_viscosity)
        end
        heatmap!(axs[k], fip.Omega.X, fip.Omega.Y, plotparam,
            colormap = cmaps[k], colorrange = clims[k])
    end

    cticks = [(50e3:50e3:250e3, num2latexstring.(50:50:250)),
        (19:23, num2latexstring.(19:23))]
    Colorbar(fig[2, 1:2], colormap = cmaps[1], colorrange = clims[1], vertical = false,
        width = Relative(0.6), ticks = cticks[1], label = clabels[1])
    Colorbar(fig[2, 3:nf], colormap = cmaps[nf], colorrange = clims[nf], vertical = false,
        width = Relative(0.6), ticks = cticks[2], label = clabels[2])

    titles = [L"(%$letter) $\,$" for letter in ["a", "b", "c", "d"]]
    [axs[k].title = titles[k] for k in 1:nf]
    [axs[k].xlabel = L"$x \: (10^3 \, \mathrm{km})$" for k in 1:nf]
    axs[1].ylabel = L"$y \: (10^3 \, \mathrm{km})$"
    # [axs[k].xticklabelsvisible = false for k in 2:nf]
    [axs[k].yticksvisible = false for k in 2:nf-1]
    [axs[k].yticklabelsvisible = false for k in 2:nf]
    axs[nf].yaxisposition = :right
    save("../publication_v1.0/plots/test3/gaussians_$suffix.png", fig)
    save("../publication_v1.0/plots/test3/gaussians_$suffix.pdf", fig)

end

main(7)
