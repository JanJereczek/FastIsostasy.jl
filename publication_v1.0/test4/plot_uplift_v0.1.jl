using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

function load_3D_results(case, N)
    @load "../data/test4/ICE6G/$case-N=$N.jld2" t fip Hitp Hice_vec deltaH
    return t, fip, Hitp, Hice_vec, deltaH
end

case = "3D-interactivesl=false-maxdepth=300000.0-nlayers=3-ICE6G_D"
N = 350
t, fip, Hitp, Hice_vec, deltaH = load_3D_results(case, N)


cmapH = cgrad([:red4, :firebrick1, :lightsalmon, :white, :cornflowerblue])
cmapu = cgrad([:magenta, :white, :cyan, :cornflowerblue, :royalblue1, :royalblue3, :midnightblue])
cmapdudt = cgrad([:grey20, :grey60, :honeydew1, :papayawhip, :khaki1, :sandybrown,
    :firebrick1, :magenta, :midnightblue, :turquoise1, :turquoise2])
H_opts = (colormap = cmapH, colorrange = (-1.5e3, 0.5e3))
u_opts = (colormap = cmapu, colorrange = (-100, 500))
dudt_opts = (colormap = cmapdudt, colorrange = (-3.5, 10.5))
Hticks = (-1500:500:500, latexify(-1500:500:500))
uticks = (-100:100:500, latexify(-100:100:500))
dudtticks = (-4:2:10, latexify(-4:2:10))

Nx, Ny = fip.Omega.Nx, fip.Omega.Ny
xx, yy = 20:Nx-20, 40:Ny-40
fig = Figure(size = (1800, 700), fontsize = 32)
axs = [Axis(fig[1, j], aspect = DataAspect()) for j in 1:3]
[hidedecorations!(ax) for ax in axs]

klgm = argmin( (seconds2years.(fip.out.t) .+ 26e3) .^ 2 )
kpi = length(fip.out.t)-2 # argmin( (seconds2years.(fip.out.t) ) .^ 2 )

hm = heatmap!(axs[1], fip.out.Hice[kpi][xx, yy] - fip.out.Hice[klgm][xx, yy]; H_opts...)
Colorbar(fig[2, 1], hm, vertical = false, flipaxis = false, ticks = Hticks,
    width = Relative(0.6), label = L"Ice thickness anomaly (m) $\,$")

ulgm = fip.out.u[klgm][xx, yy] + fip.out.ue[klgm][xx, yy]
upi = fip.out.u[kpi][xx, yy] + fip.out.ue[kpi][xx, yy]
hm = heatmap!(axs[2], upi - ulgm; u_opts...)
Colorbar(fig[2, 2], hm, vertical = false, flipaxis = false, ticks = uticks,
    width = Relative(0.6), label = L"Displacement anomaly (m) $\,$")

dudt_ms = fip.out.dudt[kpi][xx, yy] - fip.out.dudt[klgm][xx, yy]
dudt = m_per_sec2mm_per_yr.(dudt_ms) ./ 10
hm = heatmap!(axs[3], dudt; dudt_opts...)
Colorbar(fig[2, 3], hm, vertical = false, flipaxis = false, ticks = dudtticks,
    width = Relative(0.6), label = L"Displacement rate $\mathrm{(mm \, yr^{-1})}$")

rowgap!(fig.layout, 5)

fig
#=
    e1D_vec = [zeros(nm) for _ in eachindex(tlaty)]
    e3D_vec = [zeros(nm) for _ in eachindex(tlaty)]
    elaty1D_vec = [zeros(nm) for _ in eachindex(tlaty)]

    for k in eachindex(tlaty)
        k_fastiso = argmin( (t .- tlaty[k]./1e3) .^ 2 )
        uskitp = itp.(Lon, Lat, tlaty[k])
        uskitp1D = itp1D.(Lon, Lat, tlaty[k])
        usk_max[k] = maximum(abs.(uskitp))
        ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
        ufastiso1D = fip1D.out.u[k_fastiso] + fip1D.out.ue[k_fastiso]

        append!(ufi3D_vec, vec(ufastiso[mask]))
        append!(ufi1D_vec, vec(ufastiso1D[mask]))
        append!(usk_vec, vec(uskitp[mask]))
        append!(usk1D_vec, vec(uskitp1D[mask]))

        e3D = abs.(uskitp - ufastiso)
        e3D_vec[k] .= vec(e3D[mask])
        e1D = abs.(uskitp - ufastiso1D)
        e1D_vec[k] .= vec(e1D[mask])
        elaty1D = abs.(uskitp - uskitp1D)
        elaty1D_vec[k] = vec(elaty1D[mask])

        mean_error[k] = mean(e3D)
        max_error[k] = maximum(e3D)
        mean_error1D[k] = mean(e1D)
        max_error1D[k] = maximum(e1D)
    end

    tlaty_vec = vcat([fill(k, nm) for k in eachindex(tlaty)]...)
    e1D_mat = vcat(e1D_vec...)
    e3D_mat = vcat(e3D_vec...)
    elaty1D_mat = vcat(elaty1D_vec...)

    fig = Figure(size = (2150, 2300), fontsize = 40)
    ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
    ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
    axbottom = [Axis(fig[i, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9],
        i in [4:6, 7:9]]
    axs = vcat(ax_uu, ax_et, vec(axbottom))
    umax = maximum(usk_max)
    sk1D_color = :gray70

    msmax = 4
    ms = msmax .* (abs.(usk_vec) ./ umax) .^ 1.5 .+ 0.4
    # rand_idx = rand(length(usk_vec)) .< pointspercentage
    scatter!(axs[1], usk_vec, ufi1D_vec,
        markersize = ms, label = L"FI1D $\,$")
    scatter!(axs[1], usk_vec, usk1D_vec,
        markersize = ms, label = L"SK1D $\,$", color = sk1D_color)
    scatter!(axs[1], usk_vec, ufi3D_vec,
        markersize = ms, label = L"FI3D $\,$")
    lines!(axs[1], -600:50, -600:50, color = :gray10, label = label = L"identity $\,$")
    axs[1].xlabel = L"$u_\mathrm{SK3D}$ (m)"
    axs[1].ylabel = L"$u_\mathrm{m}$ (m)"
    axs[1].xticks = latexticks(-600:100:100)
    axs[1].yticks = latexticks(-600:100:100)
    xlims!(axs[1], (-600, 20))
    ylims!(axs[1], (-600, 20))
    axislegend(axs[1], position = :rb)
    axs[1].title = L"(a) $\,$"
    
    bgap = 0.25
    widthfactor = 1.2
    boxplot!(axs[2], tlaty_vec .- bgap, e1D_mat ./ umax, width = widthfactor*bgap,
        label = L"FI1D $\,$")
    boxplot!(axs[2], tlaty_vec .+ bgap, elaty1D_mat ./ umax, width = widthfactor*bgap,
        label = L"SK1D $\,$", color = sk1D_color)
    boxplot!(axs[2], tlaty_vec, e3D_mat ./ umax, width = widthfactor*bgap,
        label = L"FI3D $\,$")
    axs[2].xlabel = L"Time (kyr) $\,$"
    axs[2].ylabel = L"$e$ (1)"
    axs[2].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
    axs[2].yticks = latexticks(0:0.1:0.5)
    axislegend(axs[2], position = :lt)
    erellims = (0, 0.3)
    ylims!(axs[2], erellims)
    xlims!(axs[2], extrema(eachindex(tlaty)) .+ (-0.5, 0.5))
    axs[2].yaxisposition = :right
    axs[2].title = L"(b) $\,$"

    ulims = (-500, 500)
    elims = (-100, 100)
    u_opts = (colorrange = ulims, colormap = :PuOr)
    e_opts = (colorrange = elims, colormap = :lighttemperaturemap)
    k = argmax(max_error1D)
    k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
    uskitp = itp.(Lon, Lat, tlaty[k])
    ufastiso = fip1D.out.u[k_fastiso] + fip1D.out.ue[k_fastiso]
    heatmap!(axs[3], uskitp; u_opts...)
    heatmap!(axs[4], ufastiso; u_opts...)
    heatmap!(axs[5], uskitp - ufastiso; e_opts...)
    contour!(axs[3], mask; levels = [0.5], color = :gray10)
    contour!(axs[4], mask; levels = [0.5], color = :gray10)
    contour!(axs[5], mask; levels = [0.5], color = :gray10)
    axs[3].title = L"(c) SK3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[4].title = L"(d) FI1D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[5].title = L"(e) (SK3D - FI1D), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

    k = argmax(max_error)
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
    axs[7].title = L"(g) FI3D, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
    axs[8].title = L"(h) (SK3D - FI3D), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

    [hidedecorations!(ax) for ax in axbottom]
    Colorbar(fig[10, 2:5], hmu, label = L"Vertical displacement (m) $\,$", vertical = false,
        flipaxis = false, width = Relative(0.6), ticks = latexticks(-500:250:500))
    Colorbar(fig[10, 7:9], hme, label = L"$u_\mathrm{SK} - u_\mathrm{FI} $ (m)",
        vertical = false, flipaxis = false, width = Relative(0.8), ticks = latexticks(-100:50:100))
    save("plots/test4/$case-N=$N-mask=$masktype-final_v0.4.png", fig)
    save("plots/test4/$case-N=$N-mask=$masktype-final_v0.4.pdf", fig)
    return nothing
end

=#