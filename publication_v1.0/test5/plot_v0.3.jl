using FastIsostasy, JLD2, CairoMakie
include("../helpers.jl")

function main(; n=5)
    @load "../data/test5/n=$n.jld2" fip paraminv init_viscosity final_viscosity

    u = fip.out.u[end] + fip.out.ue[end]
    fig = Figure(resolution = (1200, 1900), fontsize = 40)
    axs = [Axis(fig[i, j], aspect = DataAspect()) for j in 1:2, i in [2:4, 5:7, 8:10]]
    [ax.aspect = DataAspect() for ax in axs[1:4]]
    [hidedecorations!(ax) for ax in axs[1:4]]
    logetalims = (19.5, 21)

    N = 2^n
    i1, i2 = Int(round(0.1 * N)), Int(round(0.6 * N))
    j1, j2 = Int(round(0.1 * N)), Int(round(0.6 * N))
    opts_eta = (colormap = cgrad(:jet, rev = true), colorrange = logetalims)
    dispmap = cgrad([:purple4, :purple1, :orchid1, :white])
    opts_u = (colormap = dispmap, colorrange = (-300, 0))
    f1 = fip.out.u[end] + fip.out.ue[end]
    f2 = log10.(fip.p.effective_viscosity)
    f3 = log10.(init_viscosity)
    f4 = log10.(final_viscosity)
    heatmap!(axs[1], f1[i1:i2, j1:j2]; opts_u...)
    heatmap!(axs[2], f2[i1:i2, j1:j2]; opts_eta...)
    heatmap!(axs[3], f3[i1:i2, j1:j2]; opts_eta...)
    heatmap!(axs[4], f4[i1:i2, j1:j2]; opts_eta...)

    Colorbar(fig[1, 1], vertical = false, width = Relative(0.8),
        ticks = latexticks(-300:100:100),
        label = L"Vertical displacement (m) $\,$"; opts_u...)
    Colorbar(fig[1, 2], vertical = false, width = Relative(0.8),
        ticks = latexticks(logetalims[1]:0.5:logetalims[2]),
        label = L"$\mathrm{log}_{10}$-viscosity (Pa s)"; opts_eta...)

    lw = 5
    ms = 25
    etavec = vec(log10.(fip.p.effective_viscosity[paraminv.data.idx]))
    grays = [:gray80, :gray60, :gray40, :gray20]
    steps = [1, 2, 5, 15]
    scatter!(axs[5], etavec, paraminv.out[1],
        color = (opts_eta.colormap[6] + opts_eta.colormap[7])/2, label = L"$m=0$")
    for i in eachindex(steps)
        nn = steps[i]
        scatter!(axs[5], etavec, paraminv.out[nn+1], color = grays[i], label = L"$m=%$nn$")
    end
    lines!(axs[5], vec(log10.(fip.p.effective_viscosity)),
        vec(log10.(fip.p.effective_viscosity)), color = :black)

    lines!(axs[6], paraminv.error, linewidth = lw)
    scatter!(axs[6], eachindex(paraminv.error)[steps], paraminv.error[steps], color = grays,
        markersize = ms)
    
    axs[5].xlabel = L"Ground truth $\mathrm{log_{10}}(\eta)$"
    axs[5].ylabel = L"Estimation $\mathrm{log_{10}}(\eta_m)$"
    axs[5].aspect = AxisAspect(1)
    # axs[6].yaxisposition = :right
    axs[5].xticks = latexticks(Union{Float64, Int}[20, 20.5, 21])
    axs[5].yticks = latexticks(Union{Float64, Int}[19.5, 20, 20.5, 21])
    xlims!(axs[5], logetalims)
    ylims!(axs[5], logetalims)
    # axislegend(axs[5], position = :rb)
    Legend(fig[11, :], axs[5], nbanks = 5, colgap = 40, markersize = 80)

    axs[6].xlabel = L"UKI iteration index $m$"
    axs[6].ylabel = L"Total displacemnt error (m) $\,$"
    # axs[5].xaxisposition = :top
    axs[6].yaxisposition = :right
    axs[6].aspect = AxisAspect(1)
    axs[6].yscale = log10
    axs[6].xticks = latexticks(0:5:15)
    axs[6].yticks = (10 .^ (1:4), [L"$10^{%$e}" for e in 1:4])

    l = ["a", "b", "c", "d", "e", "f"]
    [axs[k].title = L"(%$(l[k])) $\,$" for k in eachindex(l)]

    rowgap!(fig.layout, 20)
    # rowgap!(fig.layout, 1, -100)
    # rowgap!(fig.layout, 2, -50)
    # rowgap!(fig.layout, 3, -100)
    colgap!(fig.layout, 20)

    save("plots/test5/n=$(n)_v0.3.pdf", fig)
end

main()