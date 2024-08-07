using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles, NCDatasets
include("../helpers_plot.jl")
include("../helpers_computation.jl")
include("../../test/helpers/plot.jl")

function get_denseoutput_fastiso(fastiso_files::Vector)
    u_plot = []
    for file in fastiso_files
        @load "../data/test3/$file" fip
        @show file
        @show fip.out.computation_time
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

function get_ncdisplacement(fastiso_file::String)
    ds = NCDataset("../data/test3/$fastiso_file.nc", "r")
    t = copy(ds["t"][:])
    u = ds["u"][:, :, :] + ds["ue"][:, :, :]
    close(ds)
    return t, u
end

function mainplot(n)
    N = 2^n
    kernel = "cpu"
    suffix = "Nx$(N)_Ny$(N)_dense_premparams" # $(kernel)_

    seakon_files = ["E0L4V4", "E0L0V1"]
    fastiso_files = ["ref_$suffix.jld2", "no_litho_$suffix.jld2"]
    elims = (-20, 45)

    u_fastiso, Omega = get_denseoutput_fastiso(fastiso_files)
    t_elra, u_elra = get_ncdisplacement("elra_Nx$(N)_Ny$(N)")

    idx, r = indices_latychev2023_indices("../data/Latychev/$(seakon_files[1])", -1, 3e6)
    r .*= 1e3
    u_3DGIA = [load_latychev_gaussian("../data/Latychev/$file", idx) for file in seakon_files]

    n1, n2 = size(u_fastiso[1][1])
    slicex, slicey = n1÷2:n1, n2÷2
    x = Omega.X[slicex, slicey]

    tvec = vcat(0:1:5, 10:5:50)
    labels = [L"t = %$(tvec[k]) kyr $\,$" for k in eachindex(tvec)]
    cmap = cgrad(janjet, length(labels), categorical = true)
    lw = 3
    
    ytvisible = [true, false, false, true]
    ytlabelsvisible = [true, false, false, true]
    yaxpos = [:left, :left, :left, :right]

    xtvisible = [false, true, true]
    xtlabelsvisible = [false, true, true]

    lw = 3
    ms = 20
    elra_color = :red
    lvelva_color = :orange
    max_opts = (marker = :utriangle, markersize = ms, linewidth = lw)
    mean_opts = (marker = :circle, markersize = ms, linewidth = lw,
        linestyle = Linestyle(collect(0.0:0.7:2.1)))

    fig = Figure(size = (1000, 1200), fontsize = 34)
    ii = [3:6, 7:9, 10:12]
    nrows, ncols = 3, length(u_3DGIA)
    axs = [Axis(fig[ii[i], j],
        yticksvisible = ytvisible[j], yticklabelsvisible = ytlabelsvisible[j],
        yaxisposition = yaxpos[j], xticksvisible = xtvisible[i],
        xticklabelsvisible = xtlabelsvisible[i]) for j in 1:ncols, i in 1:nrows]

    elra_max_error = fill(Inf, length(tvec))
    elra_mean_error = fill(Inf, length(tvec))
    lvelva_max_error = fill(Inf, length(tvec))
    lvelva_mean_error = fill(Inf, length(tvec))

    sparse_idx = vcat(1:6, 7:2:15)
    for j in eachindex(u_3DGIA)
        poly!(axs[j], Point2f[(0, 0), (1e6, 0), (1e6, 1e3/25), (0, 1e3/25)],
            color = :skyblue1)

        umax = maximum(abs.(u_3DGIA[j]))
        for i in eachindex(u_fastiso[j])
            itp = linear_interpolation(r, u_3DGIA[j][:, i], extrapolation_bc = Flat())
            elra_diff = itp.(x) - u_elra[slicex, slicey, i+1]
            diff = itp.(x) - u_fastiso[j][i][slicex, slicey]
            if i in sparse_idx
                lines!(axs[j], r, u_3DGIA[j][:, i], color = cmap[i],
                    linewidth = lw, linestyle = :dash)
                lines!(axs[j], x, u_fastiso[j][i][slicex, slicey],
                    color = cmap[i], linewidth = lw, label = labels[i])
                lines!(axs[j+2], x, diff, color = cmap[i], linewidth = lw)
            end
            
            elra_max_error[i] = maximum(abs.(elra_diff)/umax)
            elra_mean_error[i] = mean(abs.(elra_diff)/umax)
            lvelva_max_error[i] = maximum(abs.(diff)/umax)
            lvelva_mean_error[i] = mean(abs.(diff)/umax)
        end

        # scatterlines!(axs[j+4], eachindex(tvec), elra_max_error,
        #     label = L"$\hat{e}_\mathrm{ELRA}$", color = elra_color; max_opts...)
        # scatterlines!(axs[j+4], eachindex(tvec), lvelva_mean_error,
        #     label = L"$\bar{e}_\mathrm{ELRA}$", color = elra_color; mean_opts...)
            
        scatterlines!(axs[j+4], eachindex(tvec), lvelva_max_error,
            label = L"$\hat{e}_\mathrm{LV-ELVA}$", color = lvelva_color; max_opts...)  
        scatterlines!(axs[j+4], eachindex(tvec), lvelva_mean_error,
            label = L"$\bar{e}_\mathrm{LV-ELVA}$", color = lvelva_color; mean_opts...)
    end

    poly!(axs[3], Point2f[(0, 1e8), (1e6, 1e8), (1e6, 1e8), (0, 1e8)],
        color = :skyblue1, label = L"ice (height scaled 1:25) $\,$")

    hlines!(axs[3], [1e3], color = :gray20, label = L"Seakon $\,$", linestyle = :dash,
        linewidth = lw, )
    hlines!(axs[3], [1e3], color = :gray20, label = L"LV-ELVA $\,$",
        linewidth = lw)
    # Legend(fig[1, 2:end-1], axs[3], nbanks = 2, framevisible = false,
    #     linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40)
    Legend(fig[1, :], axs[3], nbanks = 4, framevisible = false, colgap = 40,
        linepoints = [Point2f(0, 0.5), Point2f(2, 0.5)], patchlabelgap = 40,
        polypoints = [Point2f(0, -0.5), Point2f(2, -0.5), Point2f(2, 1.5), Point2f(0, 1.5)])

    Legend(fig[2, :], axs[1], nbanks = 4,
        linepoints = [Point2f(0, 0.5), Point2f(1.2, 0.5)], patchlabelgap = 10)
    Legend(fig[13, 2:end-1], axs[5], nbanks = 4, colgap = 50, patchsize = (30, 30))

    latexify(x) = ( x, [L"%$xi $\,$" for xi in x] )
    etks = latexify(-20:20:50)
    reletks = latexify(0.0:0.1:0.5)
    utks = latexify(-250:50:50)
    rtks = latexify(round.(-1e6:1e6:2e6))

    sparseticks = [L"%$(t) $\,$" for t in tvec]
    # for i in eachindex(sparseticks)
    #     if !(i in sparse_idx)
    #         sparseticks[i] = L""
    #     end 
    # end
    vsparse_idx = [2, 4, 6, 7, 9, 11, 13, 15]
    vsparse_bool = [i in [2, 4, 6, 7, 9, 11, 13, 15] for i in eachindex(tvec)]
    ttks = (collect(eachindex(tvec))[vsparse_bool], [L"%$(tvec[k]) $\,$" for k in vsparse_idx])

    titles = [L"(%$letter) $\,$" for letter in ["a", "b", "c", "d"]]
    [axs[j].title = titles[j] for j in 1:ncols]
    [xlims!(axs[j], (0, 2.5e6)) for j in 1:2*ncols]

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
    axs[5].xticklabelrotation = π / 2
    axs[6].xticklabelrotation = π / 2

    axs[1].ylabel = L"$u$ (m)"
    axs[3].ylabel = L"$u_\mathrm{SK} - u_\mathrm{FI}$ (m)"
    axs[5].ylabel = L"$e$ (1)"

    [ylims!(axs[j], (-300, 50)) for j in 1:ncols]
    [ylims!(axs[j], elims) for j in ncols+1:2*ncols]
    [ylims!(axs[j], (-0.01, 0.2)) for j in 2*ncols+1:3*ncols]
    rowgap!(fig.layout, 10)
    rowgap!(fig.layout, 1, 40)
    rowgap!(fig.layout, 2, 50)

    figfile = "plots/test3/test3_supplementary_N=$(N)_0.3"
    save("$figfile.png", fig)
    save("$figfile.pdf", fig)
    println(figfile)
end

global include_elastic = true
mainplot(7)