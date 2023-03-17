push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

function get_spada()
    prefix ="data/test2/Spada/"
    cases = ["u_cap", "u_disc", "dudt_cap", "dudt_disc", "n_cap", "n_disc"]
    snapshots = ["0", "1", "2", "5", "10", "inf"]
    data = Dict{String, Vector{Matrix{Float64}}}()
    for case in cases
        tmp = Matrix{Float64}[]
        for snapshot in snapshots
            fname = string(prefix, case, "_", snapshot, ".csv")
            append!(tmp, [readdlm(fname, ',', Float64)])
        end
        data[case] = tmp
    end
    return data
end

function slice_spada(
    Omega::ComputationDomain,
    t_vec::AbstractVector{T},
    t_plot::AbstractVector{T},
    vars,
    labels,
    xlabels,
    ylabels,
) where {T<:AbstractFloat}

    ncases = length(vars)
    data = get_spada()
    keys = ["u_disc", "u_cap", "dudt_disc", "dudt_cap", "n_disc", "n_cap"]

    fig = Figure(resolution=(1200, 1000), fontsize = 20)
    nrows, ncols = 3, 2
    axs = [Axis(
        fig[i, j],
        title = labels[(i-1)*ncols + j],
        xlabel = xlabels[(i-1)*ncols + j],
        ylabel = ylabels[(i-1)*ncols + j],
        xminorticks = IntervalsBetween(5),
        yminorticks = IntervalsBetween(2),
        xminorgridvisible = true,
        yminorgridvisible = true,
        xticklabelsvisible = i == nrows ? true : false,
        yticklabelsvisible = j == 1 ? true : false,
    ) for j in 1:ncols, i in 1:nrows]
    colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]

    for i in 1:ncases
        U = vars[i]
        nt = length(U)
        n1, n2 = size(U[1])
        slicey, slicex = Int(round(n1/2)), Int(round(n2/2))
        theta = rad2deg.( Omega.Theta[slicey, slicex:end] )

        bm_data = data[keys[i]]
        for k in eachindex(bm_data)
            lines!(
                axs[i],
                bm_data[k][:, 1],
                bm_data[k][:, 2],
                color = colors[k],
                linestyle = :dash,
            )
        end

        for l in eachindex(t_plot)
            t = t_plot[l]
            k = argmin( (t_vec .- t) .^ 2 )
            tyr = Int(round( seconds2years(t) ))
            lines!(
                axs[i],
                theta,
                U[k][slicey, slicex:end],
                label = L"$t = %$tyr $ yr",
                color = colors[l],
            )
        end
        if i <= 1*ncols
            ylims!(axs[i], (-450, 50))
        elseif i <= 2*ncols
            ylims!(axs[i], (-85, 10))
        elseif i <= 3*ncols
            ylims!(axs[i], (-5, 50))
        end
        xlims!(axs[i], (0, 15))
    end
    axislegend(axs[1], position = :rb)
    return fig
end


function main(
    case::String,       # Choose between viscoelastic and purely viscous response.
    n::Int;             # 2^n cells on domain (1)
    kernel = "cpu",
)

    N = 2^n
    suffix = "N$(N)_$(kernel)"
    sol_disc = load("data/test2/disc_$suffix.jld2")
    sol_cap = load("data/test2/cap_$suffix.jld2")
    res_disc = sol_disc["results"]
    res_cap = sol_cap["results"]

    if case == "viscoelastic"
        plotvars = [
            res_disc.elastic + res_disc.viscous,
            res_cap.elastic + res_cap.viscous,
            [m_per_sec2mm_per_yr.(dudt) for dudt in res_disc.displacement_rate],
            [m_per_sec2mm_per_yr.(dudt) for dudt in res_cap.displacement_rate],
            res_disc.geoid,
            res_cap.geoid,
        ]
    elseif case == "viscous"
        plotvars = [
            res_disc.viscous,
            res_cap.viscous,
            [m_per_sec2mm_per_yr.(dudt) for dudt in res_disc.displacement_rate],
            [m_per_sec2mm_per_yr.(dudt) for dudt in res_cap.displacement_rate],
            res_disc.geoid,
            res_cap.geoid,
        ]
    end
    
    labels = [
        L"Disc load $\,$",
        L"Cap load $\,$",
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
        sol_disc["Omega"],
        res_disc.t_out, t_plot,
        plotvars,
        labels, xlabels, ylabels,
    )
    plotname = "test2/$(case)_$suffix"
    save("plots/$plotname.png", response_fig)
    save("plots/$plotname.pdf", response_fig)
end

cases = ["viscoelastic", "viscous"]
for case in cases[2:2]
    for n in 6:6
        main(case, n, kernel = "cpu")
    end
end