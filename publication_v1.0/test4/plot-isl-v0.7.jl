push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

function load_3D_results(case, N)
    path = joinpath(@__DIR__, "../../data/test4/ICE6G/$case-N=$N.jld2")
    @load "$path" t fip
    return t, fip
end

function convertfip()
    N = 150
    case = "3D-interactivesl=true-maxdepth=300000.0"
    path = joinpath(@__DIR__, "../../data/test4/ICE6G/$case-N=$N.jld2")
    @load "$path" fip
    ncpath = joinpath(@__DIR__, "../../data/test4/ICE6G/$case-N=$N.nc")
    savefip(ncpath, fip)
end

function main(N; masktype="lgm")

    # case_f = "3D-interactivesl=false-maxdepth=300000.0"
    # case_i = "3D-interactivesl=true-maxdepth=300000.0"
    case_f = "3D-interactivesl=false-bsl=external"
    case_i = "3D-interactivesl=true-bsl=external"
    tf, fip_f = load_3D_results(case_f, N)
    t, fip = load_3D_results(case_i, N)
    (_, _, tlaty), _, itp = load_latychev2023_ICE6G(case = "3D")
    (_, _, _), _, itp1D = load_latychev2023_ICE6G(case = "1D")
    (_, _, _), _, Hice_itp = load_ice6gd()

    tlaty = tlaty[vcat(1:11, [13, 15, 16, 17, 18, 20, 22, 24])]

    FI_fmean = fill(Inf, length(tlaty))
    FI_fmax = fill(Inf, length(tlaty))
    SK1Dmean = fill(Inf, length(tlaty))
    SK1Dmax = fill(Inf, length(tlaty))
    FI3Dmean = fill(Inf, length(tlaty))
    FI3Dmax = fill(Inf, length(tlaty))
    usk_max = fill(Inf, length(tlaty))

    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = false)
    Lon, Lat = Omega.Lon, Omega.Lat

    Hice = [Hice_itp.(Lon, Lat, tk/1e3) for tk in tlaty]
    kmax = argmax([mean(Hice[k]) for k in eachindex(Hice)])
    mask = Hice[kmax] .> 1
    nm = sum(mask)
    maskratio = nm / prod(size(mask))

    ufi_f_vec = Float64[]
    ufi3D_vec = Float64[]
    usk1D_vec = Float64[]
    usk_vec = Float64[]

    for k in eachindex(tlaty)
        k_fastiso = argmin( (t .- tlaty[k]./1e3) .^ 2 )
        uskitp = itp.(Lon, Lat, tlaty[k])
        uskitp1D = itp1D.(Lon, Lat, tlaty[k])
        ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
        ufastiso_f = fip_f.out.u[k_fastiso] + fip_f.out.ue[k_fastiso]

        append!(ufi3D_vec, vec(ufastiso[mask]))
        append!(ufi_f_vec, vec(ufastiso_f[mask]))
        append!(usk_vec, vec(uskitp[mask]))
        append!(usk1D_vec, vec(uskitp1D[mask]))

        e3D = abs.(uskitp - ufastiso)
        e_f = abs.(uskitp - ufastiso_f)
        elaty1D = abs.(uskitp - uskitp1D)

        usk_max[k] = maximum(abs.(uskitp))
        FI3Dmean[k] = mean(e3D)
        FI3Dmax[k] = maximum(e3D)
        FI_fmean[k] = mean(e_f)
        FI_fmax[k] = maximum(e_f)
        SK1Dmean[k] = mean(elaty1D)
        SK1Dmax[k] = maximum(elaty1D)
    end

    fig = Figure(resolution = (2150, 2300), fontsize = 40)
    ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
    ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
    axbottom = [Axis(fig[i, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9],
        i in [4:6, 7:9]]
    axs = vcat(ax_uu, ax_et, vec(axbottom))
    umax = maximum(usk_max)
    sk1D_color = :gray50

    msmax = 4
    ms = msmax .* (abs.(usk_vec) ./ umax) .^ 1.5 .+ 0.4
    # rand_idx = rand(length(usk_vec)) .< pointspercentage
    scatter!(axs[1], usk_vec, ufi_f_vec,
        markersize = ms, label = L"FI fixed mask (FM) $\,$")
    scatter!(axs[1], usk_vec, usk1D_vec,
        markersize = ms, label = L"SK1D $\,$", color = sk1D_color)
    scatter!(axs[1], usk_vec, ufi3D_vec,
        markersize = ms, label = L"FI active mask (AM) $\,$")
    lines!(axs[1], -600:50, -600:50, color = :gray10, label = label = L"identity $\,$")
    axs[1].xlabel = L"$u_\mathrm{SK3D}$ (m)"
    axs[1].ylabel = L"$u_\mathrm{FI3D}$ (m)"
    axs[1].xticks = latexticks(-600:100:100)
    axs[1].yticks = latexticks(-600:100:100)
    xlims!(axs[1], (-600, 20))
    ylims!(axs[1], (-600, 20))
    axislegend(axs[1], position = :rb)
    axs[1].title = L"(a) $\,$"
    
    bwidth = 0.3
    bgap = bwidth * 0.9
    
    barplot!(axs[2], eachindex(tlaty) .- bgap, FI_fmax ./ umax,
        width = bwidth, label = L"max FI FM $\,$", color = :dodgerblue1)
    barplot!(axs[2], eachindex(tlaty) .- bgap, FI_fmean ./ umax,
        width = bwidth, label = L"mean FI FM$\,$", color = :dodgerblue3)

    barplot!(axs[2], eachindex(tlaty), SK1Dmax ./ umax,
        width = bwidth, label = L"max SK1D $\,$", color = :gray60)
    barplot!(axs[2], eachindex(tlaty), SK1Dmean ./ umax,
        width = bwidth, label = L"mean SK1D $\,$", color = :gray40)
        
    barplot!(axs[2], eachindex(tlaty) .+ bgap, FI3Dmax ./ umax,
        width = bwidth, label = L"max FI3D AM $\,$", color = :orange)
    barplot!(axs[2], eachindex(tlaty) .+ bgap, FI3Dmean ./ umax,
        width = bwidth, label = L"mean FI3D AM $\,$", color = :darkorange)

    axs[2].xlabel = L"Time (kyr) $\,$"
    axs[2].ylabel = L"$e$ (1)"
    axs[2].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
    axs[2].yticks = latexticks(0:0.1:0.5)
    axislegend(axs[2], position = :rt, nbanks = 1)
    erellims = (0, 0.3)
    ylims!(axs[2], erellims)
    xlims!(axs[2], extrema(eachindex(tlaty)) .+ (-0.5, 0.5))
    axs[2].yaxisposition = :right
    axs[2].title = L"(b) $\,$"

    ulims = (-500, 500)
    elims = (-100, 100)
    u_opts = (colorrange = ulims, colormap = :PuOr)
    e_opts = (colorrange = elims, colormap = :lighttemperaturemap)
    k = argmax(FI_fmax)
    k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
    uskitp = itp.(Lon, Lat, tlaty[k])
    ufastiso = fip_f.out.u[k_fastiso] + fip_f.out.ue[k_fastiso]
    heatmap!(axs[3], uskitp; u_opts...)
    heatmap!(axs[4], ufastiso; u_opts...)
    heatmap!(axs[5], uskitp - ufastiso; e_opts...)
    contour!(axs[3], mask; levels = [0.5], color = :gray10)
    contour!(axs[4], mask; levels = [0.5], color = :gray10)
    contour!(axs[5], mask; levels = [0.5], color = :gray10)
    axs[3].title = L"(c) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[4].title = L"(d) FI FM, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[5].title = L"(e) (SK3D - FI FM), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

    k = argmax(FI3Dmax)
    k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
    uskitp = itp.(Lon, Lat, tlaty[k])
    ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
    hmu = heatmap!(axs[6], uskitp; u_opts...)
    heatmap!(axs[7], ufastiso; u_opts...)
    hme = heatmap!(axs[8], uskitp - ufastiso; e_opts...)
    contour!(axs[6], mask; levels = [0.5], color = :gray10)
    contour!(axs[7], mask; levels = [0.5], color = :gray10)
    contour!(axs[8], mask; levels = [0.5], color = :gray10)
    axs[6].title = L"(f) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[7].title = L"(g) FI AM, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[8].title = L"(h) (SK3D - FI AM), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

    [hidedecorations!(ax) for ax in axbottom]
    Colorbar(fig[10, 2:5], hmu, label = L"Vertical displacement (m) $\,$", vertical = false,
        flipaxis = false, width = Relative(0.6), ticks = latexticks(-500:250:500))
    Colorbar(fig[10, 7:9], hme, label = L"$u_\mathrm{SK} - u_\mathrm{FI} $ (m)",
        vertical = false, flipaxis = false, width = Relative(0.8), ticks = latexticks(-100:50:100))
    save("plots/test4/$case_i-N=$N-mask=$masktype-final_v0.7.png", fig)
    return nothing
end


N = 128
main(N)

# convertfip()