push!(LOAD_PATH, "../")
using FastIsostasy, JLD2, CairoMakie
include("../helpers.jl")

function main(; n=5)
    @load "../data/test5/n=$n.jld2" fip paraminv init_viscosity final_viscosity

    u = fip.out.u[end] + fip.out.ue[end]
    fig = Figure(resolution = (1600, 1300), fontsize = 34)
    axs = [Axis(fig[1+i, j]) for i in 1:2, j in 1:3]
    [ax.aspect = DataAspect() for ax in axs[1:4]]
    [hidedecorations!(ax) for ax in axs[1:4]]
    logetalims = (19.5, 21)

    N = 2^n
    i1, i2 = Int(round(0.1 * N)), Int(round(0.6 * N))
    j1, j2 = Int(round(0.1 * N)), Int(round(0.6 * N))
    opts_eta = (colormap = cgrad(:jet, rev = true), colorrange = logetalims)
    dispmap = cgrad([:gray10, :red, :orange, :white, :cornflowerblue])
    opts_u = (colormap = dispmap, colorrange = (-300, 100))
    f1 = log10.(fip.p.effective_viscosity)
    f2 = log10.(final_viscosity)
    f3 = log10.(init_viscosity)
    f4 = fip.out.u[end] + fip.out.ue[end]
    heatmap!(axs[1], f1[i1:i2, j1:j2]; opts_eta...)
    heatmap!(axs[2], f2[i1:i2, j1:j2]; opts_eta...)
    heatmap!(axs[3], f3[i1:i2, j1:j2]; opts_eta...)
    heatmap!(axs[4], f4[i1:i2, j1:j2]; opts_u...)

    lw = 5
    ms = 25
    etavec = vec(log10.(fip.p.effective_viscosity[paraminv.data.idx]))
    grays = [:gray80, :gray60, :gray40, :gray20]
    steps = [1, 2, 5, 15]
    scatter!(axs[6], etavec, paraminv.out[1], color = (opts_eta.colormap[6] + opts_eta.colormap[7])/2)
    lines!(axs[5], paraminv.error, linewidth = lw)
    scatter!(axs[5], eachindex(paraminv.error)[steps], paraminv.error[steps], color = grays,
        markersize = ms)
    for i in eachindex(steps)
        nn = steps[i]
        scatter!(axs[6], etavec, paraminv.out[nn+1], color = grays[i])
    end
    lines!(axs[6], vec(log10.(fip.p.effective_viscosity)),
        vec(log10.(fip.p.effective_viscosity)), color = :black)

    Colorbar(fig[1, 1:2], vertical = false, width = Relative(0.4),
        ticks = latexticks(logetalims[1]:0.5:logetalims[2]),
        label = L"$\mathrm{log}_{10}$-viscosity (Pa s)"; opts_eta...)
    Colorbar(fig[4, 1:2], vertical = false, width = Relative(0.4),
        ticks = latexticks(-300:100:100), flipaxis = false,
        label = L"Vertical displacement (m) $\,$"; opts_u...)

    axs[1].xlabelvisible = true
    axs[2].xlabelvisible = true
    axs[3].xlabelvisible = true
    axs[4].xlabelvisible = true
    axs[1].xlabel = L"Ground truth $\eta$ (target)"
    axs[1].xaxisposition = :top
    axs[2].xlabel = L"Final UKI viscosity $\eta_{15}$"
    axs[2].xaxisposition = :bottom
    axs[3].xlabel = L"Initial UKI viscosity $\eta_0$"
    axs[3].xaxisposition = :top
    axs[4].xlabel = L"Observable $u(\eta, \: t = 2 \, \mathrm{kyr})$"
    axs[4].xaxisposition = :bottom
    
    axs[5].xlabel = L"UKI iteration index $\,$"
    axs[5].ylabel = L"Error w.r.t. displacement (m) $\,$"
    axs[5].xaxisposition = :top
    axs[5].yaxisposition = :right
    axs[5].aspect = AxisAspect(1)
    axs[5].yscale = log10
    axs[5].xticks = latexticks(0:5:15)
    axs[5].yticks = (10 .^ (1:4), [L"$10^{%$e}" for e in 1:4])

    axs[6].xlabel = L"Ground truth $\eta$"
    axs[6].ylabel = L"Estimation $\eta_n$"
    axs[6].aspect = AxisAspect(1)
    axs[6].yaxisposition = :right
    axs[6].xticks = latexticks(logetalims[1]+0.5:0.5:logetalims[2])
    axs[6].yticks = latexticks(logetalims[1]+0.5:0.5:logetalims[2])
    xlims!(axs[6], logetalims)
    ylims!(axs[6], logetalims)

    rowgap!(fig.layout, 1, -30)
    rowgap!(fig.layout, 3, -30)
    save("plots/test5/n=$(n)_v0.2.pdf", fig)
end

main()