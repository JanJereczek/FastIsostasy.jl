push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("../../test/helpers/compute.jl")
include("../../test/helpers/plot.jl")
include("../helpers.jl")

function slice_spada(
    Omega::ComputationDomain,
    t_vec::AbstractVector{T},
    t_plot::AbstractVector{T},
    vars,
    labels,
    xlabels,
    ylabels,
    case,
) where {T<:AbstractFloat}

    ncases = length(vars)
    (theta, t), X, Xitp = load_spada2011()
    keys = spada_cases()

    X["u_disc"] .-= X["u_disc"][:, 1]
    X["u_cap"] .-= X["u_cap"][:, 1]

    lw = 5
    fig = Figure(resolution=(1200, 1220), fontsize = 40)
    nrows, ncols = 3, 2
    rows = [1:5, 6:10, 11:15]
    axs = [Axis(
        fig[rows[i], j],
        title = labels[(i-1)*ncols + j],
        xlabel = xlabels[(i-1)*ncols + j],
        ylabel = ylabels[(i-1)*ncols + j],
        xminorticks = IntervalsBetween(5),
        yminorticks = IntervalsBetween(2),
        xminorgridvisible = true,
        yminorgridvisible = true,
        xticksvisible = i == nrows ? true : false,
        xticklabelsvisible = i == nrows ? true : false,
        yticklabelsvisible = j == 1 ? true : false,
        yaxisposition = j == 1 ? :left : :right,
    ) for j in 1:ncols, i in 1:nrows]
    # colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]
    colors = cgrad(janjet, 6, categorical = true)

    # Just for the legend
    hlines!(axs[3], [1e10], color = :gray20, label = L"FastIsostasy $\,$", linewidth = lw)
    hlines!(axs[3], [1e10], color = :gray20, linestyle = :dash, linewidth = lw,
        label = L"(Spada et al. 2011) $\,$")

    for i in 1:ncases
        U = vars[i]
        nt = length(U)
        kk = keys[i]
        n1, n2 = size(U[1])
        slicex, slicey = n1÷2:n1, n2÷2
        theta_fi = rad2deg.( Omega.Theta[slicex, slicey] )

        for j in axes(X[kk], 2)
            lines!(axs[i], theta[kk], X[kk][:, j],
                color = colors[j], linestyle = :dash, linewidth = lw)
        end

        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_vec .- t) .^ 2 )
            tkyr = Int(round( seconds2years(t) / 1e3 ))
            if i == 1
                lines!(axs[i], theta_fi, U[k][slicex, slicey],
                    color = colors[l], label = L"$t = %$tkyr $ kyr", linewidth = lw)
            else
                lines!(axs[i], theta_fi, U[k][slicex, slicey],
                    color = colors[l], linewidth = lw)
            end
        end
        if i <= 1*ncols
            ylims!(axs[i], (-420, 50))
        elseif i <= 2*ncols
            ylims!(axs[i], (-90, 10))
        elseif i <= 3*ncols
            ylims!(axs[i], (-5, 45))
        end
        xlims!(axs[i], (0, 15))
    end

    fig[16, :] = Legend(fig, axs[1], " ", framevisible = false, nbanks = 3, colgap = 30,
        height = 5)
    fig[17:18, :] = Legend(fig, axs[3], " ", framevisible = false, nbanks = 3, colgap = 50,
        linepoints = [Point2f(-1, 0.5), Point2f(1, 0.5)])
    # axislegend(axs[1], position = :rb)
    rowgap!(fig.layout, 5)
    rowgap!(fig.layout, 15, 30)
    colgap!(fig.layout, 20)

    axs[1].yticks = latexticks(-400:100:0)
    axs[3].yticks = latexticks(-80:20:0)
    axs[5].yticks = latexticks(0:20:40)

    [axs[j].xticks = latexticks(0:2.5:12.5) for j in 1:6]
    return fig
end


function main(
    case::String,       # Choose between viscoelastic and purely viscous response.
    n::Int;             # 2^n cells on domain (1)
    kernel = "cpu",
)

    N = 2^n
    suffix = "Nx=$(N)_Ny=$(N)_$(kernel)"
    filename = "disc_$suffix"
    @load "../data/test2/$filename.jld2" fip
    fipdisc = deepcopy(fip)

    filename = "cap_Nx=$(N)_Ny=$(N)_$(kernel)"
    @load "../data/test2/$filename.jld2" fip
    fipcap = deepcopy(fip)

    plotvars = [
        fipcap.out.u,
        fipdisc.out.u,
        [m_per_sec2mm_per_yr.(dudt) for dudt in fipcap.out.dudt],
        [m_per_sec2mm_per_yr.(dudt) for dudt in fipdisc.out.dudt],
        fipcap.out.geoid,
        fipdisc.out.geoid,
    ]
    
    labels = [
        L"Cap load $\,$",
        L"Cylindrical load $\,$",
        "",
        "",
        "",
        "",
    ]

    xlabels = [
        "",
        "",
        "",
        "",
        L"Colatitude $\theta$ (deg)",
        L"Colatitude $\theta$ (deg)",
    ]

    ylabels = [
        L"$u$ (m)", # Total displacement 
        "",
        L"$\dot{u} \: \mathrm{(mm \, yr^{-1}})$",   # Displacement rate 
        "",
        L"$N$ (m)", # Geoid perturbation 
        "",
    ]

    t_plot = years2seconds.([0.0, 1e3, 2e3, 5e3, 1e4, 1e5])
    response_fig = slice_spada(
        fipdisc.Omega,
        fipdisc.out.t, t_plot,
        plotvars,
        labels, xlabels, ylabels,
        case,
    )
    plotname = "test2/$suffix"
    save("plots/$(plotname)_v0.4.png", response_fig)
    save("plots/$(plotname)_v0.4.pdf", response_fig)
end

cases = ["viscous"] # "viscoelastic", 
for case in cases
    for n in 7:7
        main(case, n, kernel = "cpu")
    end
end