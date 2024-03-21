using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles
include("../helpers_computation.jl")
include("../helpers_plot.jl")
include("../../test/helpers/plot.jl")

function get_denseoutput_fastiso(fastiso_files::Vector)
    u_plot = []
    for file in fastiso_files
        @load "../data/test3/$file" fip
        if include_elastic
            push!(u_plot, fip.out.u[2:end] + fip.out.ue[2:end])
        else
            push!(u_plot, fip.out.u[2:end])
        end
    end
    @load "../data/test3/$(fastiso_files[1])" fip
    return u_plot, fip.Omega
end

function get_denseoutput_fastiso(fastiso_file::String)
    @load "../data/test3/$fastiso_file" fip
    return fip.out.u[2:end] .+ fip.out.ue[2:end]
end

function mainplot(n)
    N = 2^n
    suffix = "Nx$(N)_Ny$(N)_dense"

    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3"]
    fastiso_files = ["gaussian_lo_D_$suffix.jld2", "gaussian_hi_D_$suffix.jld2",
        "gaussian_lo_η_$suffix.jld2", "gaussian_hi_η_$suffix.jld2"]
    elims = (-30, 30)

    u_fastiso, Omega = get_denseoutput_fastiso(fastiso_files)
    u_elva = get_denseoutput_fastiso("homogeneous_Nx$(N)_Ny$(N)_dense.jld2",)
    u_elra = get_denseoutput_fastiso("elra_Nx$(N)_Ny$(N).jld2",)
    idx, r = indices_latychev2023_indices("../data/Latychev/$(seakon_files[1])", -1, 3e6)
    r .*= 1e3
    u_3DGIA = [load_latychev_gaussian("../data/Latychev/$file", idx) for file in seakon_files]

    n1, n2 = size(u_fastiso[1][1])
    slicex, slicey = n1÷2:n1, n2÷2
    x = Omega.X[slicex, slicey]

    tvec = vcat(0:1:5, 10:5:50)
    labels = [L"t = %$(tvec[k]) kyr $\,$" for k in eachindex(tvec)]
    cmap = cgrad(janjet, length(labels), categorical = true)

    ytvisible = [true, false, false, true]
    ytlabelsvisible = [true, false, false, true]
    yaxpos = [:left, :left, :left, :right]

    xtvisible = [false, true, true]
    xtlabelsvisible = [false, true, true]

    fig = Figure(size = (3200, 2000), fontsize = 58)
    ii = [3:6, 7:9, 10:12]
    axs = [Axis(fig[ii[i], j],
        yticksvisible = ytvisible[j], yticklabelsvisible = ytlabelsvisible[j],
        yaxisposition = yaxpos[j], xticksvisible = xtvisible[i],
        xticklabelsvisible = xtlabelsvisible[i]) for j in eachindex(u_3DGIA), i in 1:3]

    lw = 7
    lw2 = 5
    ms = 30
    elra_color = :red
    elva_color = :dodgerblue3
    lvelva_color = :orange
    max_opts = (marker = :utriangle, markersize = ms, linewidth = lw2)
    mean_opts = (marker = :circle, markersize = ms, linewidth = lw2,
        linestyle = Linestyle(collect(0.0:0.7:2.1)))


    elra_max_error = fill(Inf, length(tvec))
    elra_mean_error = fill(Inf, length(tvec))
    elva_max_error = fill(Inf, length(tvec))
    elva_mean_error = fill(Inf, length(tvec))
    max_error = fill(Inf, length(tvec))
    mean_error = fill(Inf, length(tvec))

    for j in eachindex(u_3DGIA)
        umax = maximum(abs.(u_3DGIA[j]))
        for i in eachindex(u_fastiso[j])
            itp = linear_interpolation(r, u_3DGIA[j][:, i], extrapolation_bc = Flat())
            lines!(axs[j], r, u_3DGIA[j][:, i], color = cmap[i],
                linewidth = lw, linestyle = :dash)
            lines!(axs[j], x, u_fastiso[j][i][slicex, slicey],
                color = cmap[i], linewidth = lw, label = labels[i])
            
            diff = itp.(x) - u_fastiso[j][i][slicex, slicey]
            lines!(axs[j+4], x, diff, color = cmap[i], linewidth = lw)
            max_error[i] = maximum(abs.(diff)/umax)
            mean_error[i] = mean(abs.(diff)/umax)

            diff_elva = itp.(x) - u_elva[i][slicex, slicey]
            elva_max_error[i] = maximum(abs.(diff_elva)/umax)
            elva_mean_error[i] = mean(abs.(diff_elva)/umax)

            diff_elra = itp.(x) - u_elra[i][slicex, slicey]
            elra_max_error[i] = maximum(abs.(diff_elra)/umax)
            elra_mean_error[i] = mean(abs.(diff_elra)/umax)
        end
        poly!(axs[j], Point2f[(0, 0), (1e6, 0), (1e6, 40), (0, 40)],
            color = :skyblue1)
        
        scatterlines!(axs[j+8], eachindex(tvec), elra_max_error, color = elra_color,
            label = L"$\hat{e}_\mathrm{ELRA}$"; max_opts...)
        scatterlines!(axs[j+8], eachindex(tvec), elra_mean_error, color = elra_color,
            label = L"$\bar{e}_\mathrm{ELRA}$"; mean_opts...)

        scatterlines!(axs[j+8], eachindex(tvec), elva_max_error, color = elva_color,
            label = L"$\hat{e}_\mathrm{ELVA}$"; max_opts...)
        scatterlines!(axs[j+8], eachindex(tvec), elva_mean_error, color = elva_color,
            label = L"$\bar{e}_\mathrm{ELVA}$"; mean_opts...)
    
        scatterlines!(axs[j+8], eachindex(tvec), max_error, color = lvelva_color,
            label = L"$\hat{e}_\mathrm{LV\text{-}ELVA}$"; max_opts...)
        scatterlines!(axs[j+8], eachindex(tvec), mean_error, color = lvelva_color,
            label = L"$\bar{e}_\mathrm{LV\text{-}ELVA}$"; mean_opts...)
    end
    poly!(axs[5], Point2f[(0, 1e8), (1e6, 1e8), (1e6, 1e8), (0, 1e8)],
        color = :skyblue1, label = L"Ice (height is 1:40) $\,$")
    hlines!(axs[5], [1e3], color = :gray20, label = L"Seakon $\,$", linestyle = :dash,
        linewidth = lw)
    hlines!(axs[5], [1e3], color = :gray20, label = L"LV-ELVA $\,$",
        linewidth = lw)
    Legend(fig[1, 2:end-1], axs[5], nbanks = 3, framevisible = false, colgap = 40,
        linepoints = [Point2f(0, 0.5), Point2f(4, 0.5)], patchlabelgap = 80,
        polypoints = [Point2f(0, -0.5), Point2f(4, -0.5), Point2f(4, 1.5), Point2f(0, 1.5)])
    Legend(fig[2, 2:end-1], axs[1], nbanks = 8,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
        # height = Relative(1.3))
    Legend(fig[13, 2:end-1], axs[9], nbanks = 6, colgap = 50, patchsize = (30, 20),
        linepoints = [Point2f(0, 0), Point2f(2, 0)], patchlabelgap = 30,
        markerpoints = [Point2f(1, 0)])

    latexify(x) = ( x, [L"%$xi $\,$" for xi in x] )
    etks = latexify(-20:10:20)
    reletks = latexify(0.0:0.1:0.5)
    utks = latexify(-250:50:50)
    rtks = latexify(round.(-1e6:1e6:2e6))
    # ttks = (0:5:15, [L"(%$k) $\,$" for k in 0:5:15])
    ttks = (eachindex(tvec)[1:2:end], [L"%$(tvec[k]) $\,$" for k in eachindex(tvec)[1:2:end]])

    titles = [L"(%$letter) $\,$" for letter in ["a", "b", "c", "d"]]
    [axs[j].title = titles[j] for j in 1:4]
    [xlims!(axs[j], (0, 3e6)) for j in 1:8]

    [axs[j].xlabel = L"$x$ (m)" for j in 5:8]
    [axs[j].xlabel = L"$t$ (kyr)" for j in 9:12]
    [axs[j].xticks = rtks for j in 5:8]
    [axs[j].xticks = ttks for j in 9:12]
    # [axs[j].xticklabelrotation = π/2 for j in 9:12]

    axs[1].yticks = utks
    axs[4].yticks = utks
    axs[5].yticks = etks
    axs[8].yticks = etks
    axs[9].yticks = reletks
    axs[12].yticks = reletks

    axs[1].ylabel = L"$u$ (m)"
    axs[5].ylabel = L"$u_\mathrm{SK} - u_\mathrm{FI}$ (m)"
    # axs[9].ylabel = L"$||u_\mathrm{SK} - u_\mathrm{FI}|| \, \mathrm{max}(u_\mathrm{SK})^{-1}$ (1)"
    axs[9].ylabel = L"$e$ (1)"

    [ylims!(axs[j], (-300, 50)) for j in 1:4]
    [ylims!(axs[j], elims) for j in 5:8]
    [ylims!(axs[j], (-0.01, 0.4)) for j in 9:12]
    rowgap!(fig.layout, 20)

    figfile = "plots/test3/test3-v0.4-N=$(N)"
    save("$figfile.pdf", fig)
end

n = 7
global include_elastic = true

mainplot(n)