push!(LOAD_PATH, "../")
using FastIsostasy, JLD2, CairoMakie
include("../helpers.jl")

function main(; n=5)
    @load "../data/test5/n=$n.jld2" fip paraminv init_viscosity final_viscosity

    u = fip.out.u[end] + fip.out.ue[end]
    fig = Figure(resolution = (1700, 1500), fontsize = 34)
    axs = [Axis(fig[1+i, j]) for j in 1:3, i in 1:2]
    [ax.aspect = DataAspect() for ax in axs[1:4]]
    [hidedecorations!(ax) for ax in axs[1:4]]

    opts_eta = (colormap = cgrad(:jet, rev = true), colorrange = (20, 21))
    opts_u = (colormap = :PuOr, colorrange = (-300, 300))
    opts_e = (colormap = :lighttemperaturemap, colorrange = (-0.1, 0.1))
    heatmap!(axs[1], log10.(fip.p.effective_viscosity); opts_eta...)
    heatmap!(axs[2], log10.(init_viscosity); opts_eta...)
    heatmap!(axs[3], fip.out.u[end] + fip.out.ue[end]; opts_u...)
    heatmap!(axs[4], log10.(final_viscosity); opts_eta...)
    # heatmap!(axs[5], log10.(fip.p.effective_viscosity) - log10.(final_viscosity);
    #     opts_e...)

    lw = 2
    etavec = vec(log10.(fip.p.effective_viscosity[paraminv.data.idx]))
    grays = [:gray80, :gray60, :gray40, :gray20]
    steps = [1, 2, 5, 15]
    lines!(axs[5], eachindex(paraminv.error), paraminv.error, linewidth = lw)
    scatter!(axs[5], eachindex(paraminv.error)[steps], paraminv.error[steps], color = grays)
    for i in eachindex(steps)
        nn = steps[i]
        println(nn)
        scatter!(axs[6], etavec, paraminv.out[nn], color = grays[i])
    end
    lines!(axs[6], vec(log10.(fip.p.effective_viscosity)),
        vec(log10.(fip.p.effective_viscosity)), color = :black)

    Colorbar(fig[1, 1:2], vertical = false, width = Relative(0.4),
        ticks = latexticks(20:0.25:21),
        label = L"$\mathrm{log}_{10}$-viscosity (Pa s)"; opts_eta...)
    Colorbar(fig[1, 3], vertical = false, width = Relative(0.8),
        ticks = latexticks(-200:100:200),
        label = L"Vertical displacement (m) $\,$"; opts_u...)
    # Colorbar(fig[4, 2], vertical = false, width = Relative(0.8),
    #     ticks = latexticks(-0.1:0.05:0.1),
    #     label = L"Difference of $\mathrm{log}_{10}$-viscosity (Pa s)"; opts_e...)

    axs[1].title = L"Ground truth $\eta$ (target)"
    axs[2].title = L"Initial UKI viscosity $\eta_0$"
    axs[3].title = L"Observable $u(\eta, \: t = 2 \, \mathrm{kyr})$ (m)"
    axs[4].title = L"Final UKI viscosity $\eta_{15}$"
    
    axs[5].title = L"Evolution of error $\,$"
    axs[5].xlabel = L"UKI iteration index $\,$"
    axs[5].ylabel = L"Error w.r.t. displacement (m) $\,$"
    axs[5].yaxisposition = :right
    axs[5].aspect = AxisAspect(1)
    # axs[5].xscale = :log10
    axs[5].yscale = log10

    axs[6].title = L"$\mathrm{log}_{10}(\eta) - \mathrm{log}_{10}(\eta_{15})$"
    axs[6].aspect = AxisAspect(1)
    xlims!(axs[6], (20, 21))

    # titles = [L"True viscosity field $\,$", L"Estimated viscosity field $\,$"]
    # cmap = cgrad(:jet, rev=true)
    # ncols = length(titles)

    # fig = Figure(resolution = (1600, 900), fontsize = 28)
    # axs = [Axis(fig[1, j], aspect = DataAspect(), title=titles[j]) for j in 1:ncols]
    # [hidedecorations!(axs[j]) for j in 1:ncols]

    # x = 1:2^n
    # heatmap!(axs[1], x, x, log10.(ground_truth), colorrange = (20, 21), colormap = cmap)
    # contour!(axs[1], x .+ 0.5, x .+ 0.5, paraminv.obs_idx, levels = [0.99], color = :white, linewidth = 5)

    # heatmap!(axs[2], x, x, log10.(paraminv.p.effective_viscosity), colorrange = (20, 21), colormap = cmap)
    # contour!(axs[2], x .+ 0.5, x .+ 0.5, paraminv.obs_idx, levels = [0.99], color = :white, linewidth = 5)

    # Colorbar(fig[2, :], colorrange = (20, 21), colormap = cmap, vertical = false, width = Relative(0.5))
    save("plots/test5/n=$(n)_v0.1.pdf", fig)
    # save("plots/test4/n=$n.png", fig)
end

main()