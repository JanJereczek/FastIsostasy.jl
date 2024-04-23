using FastIsostasy, CairoMakie, JLD2
include("../../test/helpers/compute.jl")
include("../../test/helpers/plot.jl")
include("../helpers_computation.jl")
include("../helpers_plot.jl")

function load_fip(filename)
    # println(filename)
    @load "../data/test2/$filename.jld2" fip
    return fip
end

function load_disp(filename)
    @load "../data/test2/$filename.jld2" fip
    return fip.out.u
end

function slice_spada(
    Omega::ComputationDomain,
    t_vec::AbstractVector{T},
    t_plot::AbstractVector{T},
    vars,
    labels,
    xlabels,
    ylabels,
    Hcap,
    label,
) where {T<:AbstractFloat}

    ncases = length(vars)
    (theta, t), X, Xitp = load_spada2011()
    keys = spada_cases()

    X["u_disc"] .-= X["u_disc"][:, 1]
    X["u_cap"] .-= X["u_cap"][:, 1]

    lw = 5
    fig = Figure(size=(1200, 1500), fontsize = 40)
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
        xticksvisible = i == nrows ? true : false,
        xticklabelsvisible = i == nrows ? true : false,
        yticklabelsvisible = j == 1 ? true : false,
        yaxisposition = j == 1 ? :left : :right,
    ) for j in 1:ncols, i in 1:nrows]
    # colors = [:gray80, :gray65, :gray50, :gray35, :gray20, :gray5]
    colors = cgrad(janjet, 6, categorical = true)

    bulgecolor = (:gray70, 0.5)
    poly!(axs[1], Point2f[(11, 1e3), (11, -1e3), (15, -1e3), (15, 1e3)],
        color = bulgecolor)
    poly!(axs[2], Point2f[(11.5, 1e3), (11.5, -1e3), (15, -1e3), (15, 1e3)],
        color = bulgecolor)
    
    # Just for the legend
    hlines!(axs[3], [1e10], color = :gray20, label = L"%$label $\,$", linewidth = lw)
    hlines!(axs[3], [1e10], color = :gray20, linestyle = :dash, linewidth = lw,
        label = L"Spada et al. (2011) $\,$")

    n1, n2 = size(vars[1][1])
    slicex, slicey = n1÷2:n1, n2÷2
    theta_fi = rad2deg.( Omega.Theta[slicex, slicey] )
    mean_error = zeros(length(t_plot))
    max_error = zeros(length(t_plot))

    for i in 1:ncases
        U = vars[i]
        nt = length(U)
        kk = keys[i]

        for l in eachindex(t_plot)

            lines!(axs[i], theta[kk], X[kk][:, l], color = colors[l], linestyle = :dash,
                linewidth = lw)

            t = t_plot[l]
            k = argmin( (t_vec .- t) .^ 2 )
            tkyr = Int(round( seconds2years(t) / 1e3 ))
            # itp = linear_interpolation(theta[kk], X[kk][:, l], extrapolation_bc=Flat())
            # uspada = itp.(theta_fi)

            itp = linear_interpolation(theta_fi, U[k][slicex, slicey], extrapolation_bc=Throw())
            ufi = itp.(theta[kk])
            if i == 1
                lines!(axs[i], theta_fi, U[k][slicex, slicey],
                    color = colors[l], label = L"$%$tkyr \, \mathrm{kyr}$ ", linewidth = lw)
            else
                lines!(axs[i], theta_fi, U[k][slicex, slicey],
                    color = colors[l], linewidth = lw)
            end

            # mean_error[l] = mean(abs.(U[k][slicex, slicey] - uspada))
            # max_error[l] = maximum(abs.(U[k][slicex, slicey] - uspada))

            mean_error[l] = mean(abs.(ufi - X[kk][:, l]))
            max_error[l] = maximum(abs.(ufi - X[kk][:, l]))

            # if i <= 2
            #     Uelra = addvars[i]
            #     lines!(axs[i], theta_fi, Uelra[k][slicex, slicey], color = (colors[l], 0.5),
            #         linewidth = lw)
            # end
        end
        if i <= 1*ncols
            ylims!(axs[i], (-420, 70))
        elseif i <= 2*ncols
            ylims!(axs[i], (-90, 10))
        elseif i <= 3*ncols
            ylims!(axs[i], (-5, 45))
        end
        xlims!(axs[i], (0, 15))

        println("Mean error: ", mean_error)
        println("Max error: ", max_error)
        println("------------------------")
    end

    Hcapslice = Hcap[slicex, slicey]
    Hlolim = zeros(length(Hcapslice))
    band!(axs[1], theta_fi, Hlolim, Hcapslice ./ 25, color = :skyblue1)
    poly!(axs[2], Point2f[(0, 0), (10, 0), (10, 1e3 / 25), (0, 1e3 / 25)], color = :skyblue1)

    poly!(axs[3], Point2f[(0, 1e3), (10, 1e3), (10, 2e3), (0, 2e3)], color = :skyblue1,
        strokecolor = :skyblue1, strokewidth = 3, label = L"ice (height scaled 1:25) $\,$")
    poly!(axs[3], Point2f[(1e3, 1e3), (1e3, 1e3), (1e3, 1e3), (1e3, 1e3)],
        color = bulgecolor, label = L"forebulge $\,$")

    fig[4, :] = Legend(fig, axs[1], " ", framevisible = false, nbanks = 6, colgap = 30,
        height = 5)
    fig[5, :] = Legend(fig, axs[3], " ", framevisible = false, nbanks = 2, colgap = 40,
        linepoints = [Point2f(-1, 0.5), Point2f(1, 0.5)])

    # axislegend(axs[1], position = :rb)
    rowgap!(fig.layout, 5)
    rowgap!(fig.layout, 3, -20)
    rowgap!(fig.layout, 4, -30)
    colgap!(fig.layout, 20)

    rowsize!(fig.layout, 1, Relative(0.27))
    rowsize!(fig.layout, 2, Relative(0.27))
    rowsize!(fig.layout, 3, Relative(0.27))
    rowsize!(fig.layout, 4, Relative(0.08))
    rowsize!(fig.layout, 5, Relative(0.12))

    axs[1].yticks = latexticks(-400:100:100)
    axs[3].yticks = latexticks(-80:20:0)
    axs[5].yticks = latexticks(0:20:40)

    [axs[j].xticks = latexticks(0:2.5:12.5) for j in 1:6]
    return fig
end


function main(
    case::String,       # Choose between viscoelastic and purely viscous response.
    n::Int;             # 2^n cells on domain (1)
    kernel = "cpu",
    solver = :elva,
)

    N = 2^n
    suffix = "Nx=$(N)_Ny=$(N)"
    if solver == :elva
        prefix = ""
        label = "FastIsostasy"
    elseif solver == :elra
        prefix = "elra-"
        label = "ELRA"
    end

    filename = "$(prefix)cap_$suffix"
    fipcap = load_fip(filename)
    Hcap = fipcap.out.Hice[end]

    filename = "$(prefix)disc_$suffix"
    # println(filename)
    fipdisc = load_fip(filename)

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
        Hcap,
        label,
    )
    plotname = "test2/$suffix"
    # save("plots/$(plotname)-$(prefix)v0.6.png", response_fig)
    # save("plots/$(plotname)-$(prefix)v0.6.pdf", response_fig)
end

solvers = [:elva]
for solver in solvers
    main("viscous", 7, solver = solver)
end