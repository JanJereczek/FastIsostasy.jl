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

    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3"]
    fastiso_files = ["gaussian_lo_D_$suffix.jld2", "gaussian_hi_D_$suffix.jld2",
        "gaussian_lo_η_$suffix.jld2", "gaussian_hi_η_$suffix.jld2"]
    elims = (-30, 30)

    # elseif heterogeneous == "none"
    #     seakon_files = ["E0L4V4", "E0L0V1"]
    #     fastiso_files = ["ref_$suffix.jld2", "no_litho_$suffix.jld2"]
    #     elims = (-20, 45)
    #     title1 = L"Homogeneous PREM configuration $\,$"
    #     title2 = L"No-lithosphere configuration $\,$"
    # end

    u_fastiso, Omega = get_denseoutput_fastiso(fastiso_files)
    u_elva = get_denseoutput_fastiso("homogeneous_$suffix.jld2",)
    idx, r = indices_latychev2023_indices("../data/Latychev/$(seakon_files[1])", -1, 3e6)
    r .*= 1e3
    u_3DGIA = [load_latychev_gaussian("../data/Latychev/$file", idx) for file in seakon_files]

    n1, n2 = size(u_fastiso[1][1])
    slicex, slicey = n1÷2:n1, n2÷2
    x = Omega.X[slicex, slicey]

    tvec = vcat(0:1:5, 10:5:50)
    labels = [L"t = %$(tvec[k]) kyr $\,$" for k in eachindex(tvec)]
    cmap = cgrad(janjet, length(labels), categorical = true)
    bgap = 0.2
    bwidth = 1.95 * bgap

    ytvisible = [true, false, false, true]
    ytlabelsvisible = [true, false, false, true]
    yaxpos = [:left, :left, :left, :right]

    xtvisible = [false, true, true]
    xtlabelsvisible = [false, true, true]

    fig = Figure(resolution = (3200, 2000), fontsize = 58)
    ii = [3:6, 7:9, 10:12]
    axs = [Axis(fig[ii[i], j],
        yticksvisible = ytvisible[j], yticklabelsvisible = ytlabelsvisible[j],
        yaxisposition = yaxpos[j], xticksvisible = xtvisible[i],
        xticklabelsvisible = xtlabelsvisible[i]) for j in eachindex(u_3DGIA), i in 1:3]

    lw = 7
    max_error = fill(Inf, length(tvec))
    mean_error = fill(Inf, length(tvec))
    elva_max_error = fill(0.5, length(tvec))
    elva_mean_error = fill(0.5, length(tvec))
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
        end
        barplot!(axs[j+8], eachindex(tvec) .- bgap, elva_max_error,
            width = bwidth, label = L"FI1D max $\,$")
        barplot!(axs[j+8], eachindex(tvec) .- bgap, elva_mean_error,
            width = bwidth, label = L"FI1D mean $\,$", color = :gray75)
        
        barplot!(axs[j+8], eachindex(tvec) .+ bgap, max_error,
            width = bwidth, label = L"FI3D max $\,$")
        barplot!(axs[j+8], eachindex(tvec) .+ bgap, mean_error,
            width = bwidth, label = L"FI3D mean $\,$", color = :gray50)
    end

    hlines!(axs[5], [1e3], color = :gray20, label = L"Seakon $\,$", linestyle = :dash,
        linewidth = lw, )
    hlines!(axs[5], [1e3], color = :gray20, label = L"FastIsostasy $\,$",
        linewidth = lw)
    Legend(fig[1, 2:end-1], axs[5], nbanks = 2, framevisible = false,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
    Legend(fig[2, 2:end-1], axs[1], nbanks = 8,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
        # height = Relative(1.3))
    Legend(fig[13, 2:end-1], axs[9], nbanks = 4, colgap = 50, patchsize = (40, 40),
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)])

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

    figfile = "plots/test3/test3_v0.1_N=$(N)"
    save("$figfile.png", fig)
    save("$figfile.pdf", fig)
    println(figfile)
end

n = 7
heterogeneous = "none"
global include_elastic = true

mainplot(n)