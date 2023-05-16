push!(LOAD_PATH, "../")
using FastIsostasy, JLD2, CairoMakie

function main(; n=5)
    sol = load("data/test6/n=$n.jld2")
    ground_truth, paraminv = sol["ground_truth"], sol["paraminv"]
    # ground_truth, paraminv = sol["ground_truth"], sol["paraminv"]

    titles = [L"True viscosity field $\,$", L"Estimated viscosity field $\,$"]
    cmap = cgrad(:jet, rev=true)
    ncols = length(titles)

    fig = Figure(resolution = (1600, 900), fontsize = 28)
    axs = [Axis(fig[1, j], aspect = DataAspect(), title=titles[j]) for j in 1:ncols]
    [hidedecorations!(axs[j]) for j in 1:ncols]

    x = 1:2^n
    heatmap!(axs[1], x, x, log10.(ground_truth), colorrange = (20, 21), colormap = cmap)
    contour!(axs[1], x .+ 0.5, x .+ 0.5, paraminv.obs_idx, levels = [0.99], color = :white, linewidth = 5)

    heatmap!(axs[2], x, x, log10.(paraminv.p.effective_viscosity), colorrange = (20, 21), colormap = cmap)
    contour!(axs[2], x .+ 0.5, x .+ 0.5, paraminv.obs_idx, levels = [0.99], color = :white, linewidth = 5)

    Colorbar(fig[2, :], colorrange = (20, 21), colormap = cmap, vertical = false, width = Relative(0.5))
    save("plots/test6/n=$n.pdf", fig)
    save("plots/test6/n=$n.png", fig)
end

main()