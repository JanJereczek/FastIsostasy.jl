push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles
include("../helpers.jl")
include("../../test/helpers/plot.jl")

function indices_latychev2023_indices(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]
    return idx, x
end

function load_latychev_gaussian(dir::String, idx)
    files = readdir(dir)
    nr = size( readdlm(joinpath(dir, files[1]), ','), 1 )
    u = zeros(nr, length(idx))
    for i in eachindex(files)
        u[:, i] = readdlm(joinpath(dir, files[i]), ',')[:, 2]
    end
    return u[idx, :]
end

function get_denseoutput_fastiso(fastiso_files::Vector)
    u_plot = []
    for file in fastiso_files
        @load "../data/test3/$file" fip
        if include_elastic
            push!(u_plot, fip.out.u + fip.out.ue)
        else
            push!(u_plot, fip.out.u)
        end
    end
    @load "../data/test3/$(fastiso_files[1])" fip
    return u_plot, fip.Omega
end

function get_denseoutput_fastiso(fastiso_file::String)
    @load "../data/test3/$fastiso_file" fip
    return fip.out.u .+ fip.out.ue
end

function mainplot(n)
    N = 2^n
    kernel = "cpu"
    suffix = "$(kernel)_Nx$(N)_Ny$(N)_dense"

    seakon_files = ["E0L4V4", "E0L0V1"]
    fastiso_files = ["ref_$suffix.jld2", "no_litho_$suffix.jld2"]
    elims = (-20, 45)

    #     title1 = L"Homogeneous PREM configuration $\,$"
    #     title2 = L"No-lithosphere configuration $\,$"

    u_fastiso, Omega = get_denseoutput_fastiso(fastiso_files)
    idx, r = indices_latychev2023_indices("../data/Latychev/$(seakon_files[1])", -1, 3e6)
    r .*= 1e3
    u_3DGIA = [load_latychev_gaussian("../data/Latychev/$file", idx) for file in seakon_files]

    n1, n2 = size(u_fastiso[1][1])
    slicex, slicey = n1÷2:n1, n2÷2
    x = Omega.X[slicex, slicey]

    tvec = vcat(0:1:5, 10:5:50)
    labels = [L"t = %$(tvec[k]) kyr $\,$" for k in eachindex(tvec)]
    cmap = cgrad(janjet, length(labels), categorical = true)
    bwidth = 0.8
    lw = 3
    
    ytvisible = [true, false, false, true]
    ytlabelsvisible = [true, false, false, true]
    yaxpos = [:left, :left, :left, :right]

    xtvisible = [false, true, true]
    xtlabelsvisible = [false, true, true]

    fig = Figure(resolution = (1200, 1200), fontsize = 30)
    ii = [3:6, 7:9, 10:12]
    nrows, ncols = 3, length(u_3DGIA)
    axs = [Axis(fig[ii[i], j],
        yticksvisible = ytvisible[j], yticklabelsvisible = ytlabelsvisible[j],
        yaxisposition = yaxpos[j], xticksvisible = xtvisible[i],
        xticklabelsvisible = xtlabelsvisible[i]) for j in 1:ncols, i in 1:nrows]

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
            lines!(axs[j+2], x, diff, color = cmap[i], linewidth = lw)
            max_error[i] = maximum(abs.(diff)/umax)
            mean_error[i] = mean(abs.(diff)/umax)
        end
        barplot!(axs[j+4], eachindex(tvec), max_error,
            width = bwidth, label = L"FI max $\,$")
        barplot!(axs[j+4], eachindex(tvec), mean_error,
            width = bwidth, label = L"FI mean $\,$")
    end

    hlines!(axs[3], [1e3], color = :gray20, label = L"Seakon $\,$", linestyle = :dash,
        linewidth = lw, )
    hlines!(axs[3], [1e3], color = :gray20, label = L"FastIsostasy $\,$",
        linewidth = lw)
    Legend(fig[1, 2:end-1], axs[3], nbanks = 2, framevisible = false,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
    Legend(fig[2, :], axs[1], nbanks = 5, width = Relative(0.95),
        linepoints = [Point2f(0, 0.5), Point2f(1.2, 0.5)], patchlabelgap = 10)
    Legend(fig[13, 2:end-1], axs[5], nbanks = 4, colgap = 50, patchsize = (30, 30))

    latexify(x) = ( x, [L"%$xi $\,$" for xi in x] )
    etks = latexify(-20:10:50)
    reletks = latexify(0.0:0.1:0.5)
    utks = latexify(-250:50:50)
    rtks = latexify(round.(-1e6:1e6:2e6))
    ttks = (eachindex(tvec)[1:2:end], [L"%$(tvec[k]) $\,$" for k in eachindex(tvec)[1:2:end]])

    titles = [L"(%$letter) $\,$" for letter in ["a", "b", "c", "d"]]
    [axs[j].title = titles[j] for j in 1:ncols]
    [xlims!(axs[j], (0, 3e6)) for j in 1:2*ncols]

    [axs[j].xlabel = L"$x$ (m)" for j in ncols+1:2*ncols]
    [axs[j].xlabel = L"$t$ (kyr)" for j in 2*ncols+1:3*ncols]
    [axs[j].xticks = rtks for j in ncols+1:2*ncols]
    [axs[j].xticks = ttks for j in 2*ncols+1:3*ncols]

    axs[1].yticks = utks
    axs[2].yticks = utks
    axs[3].yticks = etks
    axs[4].yticks = etks
    axs[5].yticks = reletks
    axs[6].yticks = reletks

    axs[1].ylabel = L"$u$ (m)"
    axs[3].ylabel = L"$u_\mathrm{SK} - u_\mathrm{FI}$ (m)"
    axs[5].ylabel = L"$e$ (1)"

    [ylims!(axs[j], (-300, 50)) for j in 1:ncols]
    [ylims!(axs[j], elims) for j in ncols+1:2*ncols]
    [ylims!(axs[j], (-0.01, 0.2)) for j in 2*ncols+1:3*ncols]
    rowgap!(fig.layout, 20)
    rowgap!(fig.layout, 2, 40)

    figfile = "plots/test3/test3_supplementary_N=$(N)"
    save("$figfile.png", fig)
    save("$figfile.pdf", fig)
    println(figfile)
end

global include_elastic = true
mainplot(7)