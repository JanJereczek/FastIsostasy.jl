using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers_computation.jl")
include("../helpers_plot.jl")

function load_elva_displacement(N, isl)
    ds = NCDataset("../data/test4/ICE6G/1D-interactivesl=$isl-maskbsl=true-N=$N.nc", "r")
    u = ds["u"][:, :, :] + ds["ue"][:, :, :]
    close(ds)
    return u
end

function load_lvelva_displacement(N)
    ds = NCDataset("../data/test4/ICE6G/3D-interactivesl=true-maskbsl=true-N=$N.nc", "r")
    t = ds["t"][:]
    u = ds["u"][:, :, :] + ds["ue"][:, :, :]
    mask = ds["active mask"][:, :]
    Hice = ds["Hice"][:, :, :]
    return t, u, mask, Hice
end

function load_elra_displacement(case, N)
    ds = NCDataset("../data/test4/ICE6G/$case-interactivesl=false-bsl=external-N=$N.nc", "r")
    u = ds["u"][:, :, :] + ds["ue"][:, :, :]
    close(ds)
    return u
end

function main(case, N)
    t, u_lvelva, mask, Hice = load_lvelva_displacement(N)
    u_elva = load_elva_displacement(N, true)
    # u_elva_isl = load_elva_displacement(N, true)
    u_elra = load_elra_displacement("elra", N)
    (_, _, tlaty), _, itp = load_latychev2023_ICE6G(case = "3D")
    (_, _, _), _, itp1D = load_latychev2023_ICE6G(case = "1D")

    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = false)
    Lon, Lat = Omega.Lon, Omega.Lat
    klgm = argmax([mean(H) for H in eachslice(Hice, dims = 3)])
    lgm = ((Hice[:, :, klgm] .> 1) .|| (Omega.R .< 500e3)) .&& (Omega.Y .> -2_400e3)
    mask = mask .> 0.5
    tlaty = tlaty[vcat(1:11, [13, 15, 16, 17, 18, 20, 22, 24])]

    elra_mean = fill(Inf, length(tlaty))
    elra_max = fill(Inf, length(tlaty))
    elva_mean = fill(Inf, length(tlaty))
    elva_max = fill(Inf, length(tlaty))
    # elva_isl_mean = fill(Inf, length(tlaty))
    # elva_isl_max = fill(Inf, length(tlaty))
    lvelva_mean = fill(Inf, length(tlaty))
    lvelva_max = fill(Inf, length(tlaty))
    sk1D_mean = fill(Inf, length(tlaty))
    sk1D_max = fill(Inf, length(tlaty))
    usk_max = fill(Inf, length(tlaty))

    elra_vec = Float64[]
    elva_vec = Float64[]
    lvelva_vec = Float64[]
    usk1D_vec = Float64[]
    usk_vec = Float64[]

    for k in eachindex(tlaty)
        k_fastiso = argmin( (t .- tlaty[k]) .^ 2 )
        u_elra_k = u_elra[:, :, k_fastiso]
        u_elva_k = u_elva[:, :, k_fastiso]
        u_lvelva_k = u_lvelva[:, :, k_fastiso]
        uskitp1D = itp1D.(Lon, Lat, tlaty[k])
        uskitp = itp.(Lon, Lat, tlaty[k])

        append!(elra_vec, vec(u_elra_k[mask]))
        append!(elva_vec, vec(u_elva_k[mask]))
        append!(lvelva_vec, vec(u_lvelva_k[mask]))
        append!(usk_vec, vec(uskitp[mask]))
        append!(usk1D_vec, vec(uskitp1D[mask]))

        e_elra = abs.(uskitp - u_elra_k) # .* mask
        e_elva = abs.(uskitp - u_elva_k) # .* mask
        # e_elva_isl = abs.(uskitp - u_elva_isl[:, :, k_fastiso]) .* mask
        e_lvelva = abs.(uskitp - u_lvelva_k) # .* mask
        elaty1D = abs.(uskitp - uskitp1D) # .* mask

        usk_max[k] = maximum(abs.(uskitp))
        
        elra_mean[k] = mean(e_elra)
        elra_max[k] = maximum(e_elra)
        elva_mean[k] = mean(e_elva)
        elva_max[k] = maximum(e_elva)
        # elva_isl_mean[k] = mean(e_elva_isl)
        # elva_isl_max[k] = maximum(e_elva_isl)
        lvelva_mean[k] = mean(e_lvelva)
        lvelva_max[k] = maximum(e_lvelva)
        sk1D_mean[k] = mean(elaty1D)
        sk1D_max[k] = maximum(elaty1D)

    end

    fig = Figure(size = (2100, 2300), fontsize = 40)
    ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
    ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
    axbottom = [Axis(fig[i, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9],
        i in [4:6, 7:9]]
    axs = vcat(ax_uu, ax_et, vec(axbottom))
    umax = maximum(usk_max)
    elra_color = :red
    elva_color = :dodgerblue3
    sk1D_color = :gray50
    lvelva_color = :orange

    msmax = 4
    ms = msmax .* (abs.(usk_vec) ./ umax) .^ 1.5 .+ 0.4
    # rand_idx = rand(length(usk_vec)) .< pointspercentage
    scatter!(axs[1], usk_vec, elra_vec, markersize = ms, color = elra_color)
    scatter!(axs[1], usk_vec, elva_vec, markersize = ms, color = elva_color)
    scatter!(axs[1], usk_vec, usk1D_vec, markersize = ms, color = sk1D_color)
    scatter!(axs[1], usk_vec, lvelva_vec, markersize = ms, color = lvelva_color)
    
    scatter!(axs[1], 1e10, 1e10, markersize = 20, label = L"ELRA $\,$", color = elra_color)
    scatter!(axs[1], 1e10, 1e10, markersize = 20, label = L"ELVA $\,$", color = elva_color)
    scatter!(axs[1], 1e10, 1e10, markersize = 20, label = L"SK1D $\,$", color = sk1D_color)
    scatter!(axs[1], 1e10, 1e10, markersize = 20, label = L"LV-ELVA $\,$", color = lvelva_color)

    lines!(axs[1], -600:50, -600:50, color = :gray10, label = label = L"identity $\,$")
    axs[1].xlabel = L"$u_\mathrm{SK3D}$ (m)"
    axs[1].ylabel = L"$u_\mathrm{m}$ (m)"
    axs[1].xticks = latexticks(-600:100:100)
    axs[1].yticks = latexticks(-600:100:100)
    xlims!(axs[1], (-600, 20))
    ylims!(axs[1], (-600, 20))
    axislegend(axs[1], position = :rb, fontsize = 24)
    axs[1].title = L"(a) $\,$"
    
    lw = 5
    alpha = 1.0

    max_opts = (marker = :utriangle, markersize = 30, linewidth = lw)
    mean_opts = (marker = :circle, markersize = 30, linewidth = lw, linestyle = :dash)
    
    scatterlines!(axs[2], eachindex(tlaty), elra_max ./ umax,
        label = L"$ \hat{e}_\mathrm{ELRA}$", color = elra_color; max_opts...)
    scatterlines!(axs[2], eachindex(tlaty), elra_mean ./ umax .* alpha,
        label = L"$ \bar{e}_\mathrm{ELRA}$", color = elra_color; mean_opts...)
        
    scatterlines!(axs[2], eachindex(tlaty), elva_max ./ umax,
        label = L"$ \hat{e}_\mathrm{ELVA}$", color = elva_color; max_opts...)
    scatterlines!(axs[2], eachindex(tlaty), elva_mean ./ umax .* alpha,
        label = L"$ \bar{e}_\mathrm{ELVA}$", color = elva_color; mean_opts...)

    scatterlines!(axs[2], eachindex(tlaty), sk1D_max ./ umax,
        label = L"$ \hat{e}_\mathrm{SK1D}$", color = sk1D_color; max_opts...)
    scatterlines!(axs[2], eachindex(tlaty), sk1D_mean ./ umax .* alpha,
        label = L"$ \bar{e}_\mathrm{SK1D}$", color = sk1D_color; mean_opts...)

    scatterlines!(axs[2], eachindex(tlaty), lvelva_max ./ umax,
        label = L"$ \hat{e}_\mathrm{LV\text{-}ELVA}$", color = lvelva_color; max_opts...)
    scatterlines!(axs[2], eachindex(tlaty), lvelva_mean ./ umax .* alpha,
        label = L"$ \bar{e}_\mathrm{LV\text{-}ELVA}$", color = lvelva_color; mean_opts...)

    # scatterlines!(axs[2], eachindex(tlaty), elva_isl_max ./ umax,
    #     label = L"$ \hat{e}_\mathrm{ELVA-ISL}$", color = :purple; max_opts...)
    # scatterlines!(axs[2], eachindex(tlaty), elva_isl_mean ./ umax .* alpha,
    #     label = L"$ \bar{e}_\mathrm{ELVA-ISL}$", color = :purple; mean_opts...)

    axs[2].xlabel = L"Time (kyr) $\,$"
    axs[2].ylabel = L"$e$ (1)"
    axs[2].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
    axs[2].yticks = latexticks(0:0.1:0.5)
    # Legend(fig[10, 3:7], axs[2], nbanks = 4, fontsize = 26,
    #     patchsize = (40.0f0, 20.0f0), colgap = 25)
    axislegend(axs[2], nbanks = 2, fontsize = 24, patchsize = (40.0f0, 20.0f0),
        colgap = 20, position = :lt)
    erellims = (0, 0.24)
    ylims!(axs[2], erellims)
    xlims!(axs[2], extrema(eachindex(tlaty)) .+ (-0.5, 0.5))
    axs[2].yaxisposition = :right
    axs[2].title = L"(b) $\,$"
    axs[2].yminorticks = IntervalsBetween(5)
    axs[2].yminorgridvisible = true

    ulims = (-500, 500)
    elims = (-100, 100)
    u_opts = (colorrange = ulims, colormap = :PuOr)
    e_opts = (colorrange = elims, colormap = :lighttemperaturemap)
    k = argmax(elva_max)
    k_fastiso = argmin( (t .- tlaty[k]) .^ 2 )
    uskitp = itp.(Lon, Lat, tlaty[k])
    u_elva_k = u_elva[:, :, k_fastiso]
    heatmap!(axs[3], u_elva_k; u_opts...)
    heatmap!(axs[4], uskitp; u_opts...)
    heatmap!(axs[5], uskitp - u_elva_k; e_opts...)

    contour!(axs[3], mask; levels = [0.5], color = :gray10, linewidth = 2)
    contour!(axs[4], mask; levels = [0.5], color = :gray10, linewidth = 2)
    contour!(axs[5], mask; levels = [0.5], color = :gray10, linewidth = 2)

    contour!(axs[3], lgm; levels = [0.5], color = :gray50, linewidth = 2)
    contour!(axs[4], lgm; levels = [0.5], color = :gray50, linewidth = 2)
    contour!(axs[5], lgm; levels = [0.5], color = :gray50, linewidth = 2)

    axs[3].title = L"(c) ELVA, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[4].title = L"(d) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[5].title = L"(e) (SK3D $-$ ELVA)"

    k = argmax(lvelva_max)
    k_fastiso = argmin( (t .- tlaty[k]) .^ 2 )
    uskitp = itp.(Lon, Lat, tlaty[k])
    u_lvelva_k = u_lvelva[:, :, k_fastiso]
    heatmap!(axs[6], u_lvelva_k; u_opts...)
    hmu = heatmap!(axs[7], uskitp; u_opts...)
    hme = heatmap!(axs[8], uskitp - u_lvelva_k; e_opts...)

    contour!(axs[6], mask; levels = [0.5], color = :gray10, linewidth = 2)
    contour!(axs[7], mask; levels = [0.5], color = :gray10, linewidth = 2)
    contour!(axs[8], mask; levels = [0.5], color = :gray10, linewidth = 2)

    contour!(axs[6], lgm; levels = [0.5], color = :gray50, linewidth = 2)
    contour!(axs[7], lgm; levels = [0.5], color = :gray50, linewidth = 2)
    contour!(axs[8], lgm; levels = [0.5], color = :gray50, linewidth = 2)

    axs[6].title = L"(f) LV-ELVA, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[7].title = L"(g) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[8].title = L"(h) (SK3D $-$ LV-ELVA)"

    [hidedecorations!(ax) for ax in axbottom]
    Colorbar(fig[10, 1:3], hmu, label = L"Vertical displacement (m) $\,$", vertical = false,
        flipaxis = false, width = Relative(0.8), height = Relative(0.15),
        ticks = latexticks(-500:250:500))
    Colorbar(fig[10, 7:9], hme, label = L"Difference to SK3D (m) $\,$",
        vertical = false, flipaxis = false, width = Relative(0.8),
        height = Relative(0.15), ticks = latexticks(-100:50:100))
    
    elem_0 = LineElement(color = :transparent, linewidth = 4)
    elem_1 = LineElement(color = :gray10, linewidth = 4)
    elem_2 = LineElement(color = :gray50, linewidth = 4)
    Legend(fig[10, 4:6],
        [elem_0, elem_0, elem_1, elem_2],
        ["", "", L"Boundary of active region $\,$", L"LGM extent of AIS $\,$"],
        patchsize = (35, 35), rowgap = 10, framevisible = false)

    rowgap!(fig.layout, 9, -30.0)
    colgap!(fig.layout, 10.0)
    save("plots/test4/$case-N=$N-final_v0.8.png", fig)
    return nothing
end

case = "3D-interactivesl=true-bsl=external"
N = 350
main(case, 350)