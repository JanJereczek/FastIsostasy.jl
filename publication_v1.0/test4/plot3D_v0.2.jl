using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("laty.jl")
include("../helpers.jl")

case = "3D-interactivesl=false-ICE6G_D"
N = 280
@load "../data/test4/ICE6G/$case-N=$N.jld2" t fip Hitp Hice_vec deltaH
make_anims = false

function load_1D_results(N)
    @load "../data/test4/ICE6G/1D-interactivesl=false-ICE6G_D"*
        "-N=$N.jld2" t fip Hitp Hice_vec deltaH
    return t, fip
end

t1D, fip1D = load_1D_results(N);
tlaty, ulaty, Lon, Lat, itp = load_laty_ICE6G(case = case[1:2])

elims = (-100, 100)
mean_error = fill(Inf, length(tlaty))
max_error = fill(Inf, length(tlaty))
mean_error1D = fill(Inf, length(tlaty))
max_error1D = fill(Inf, length(tlaty))
ulaty_max = fill(Inf, length(tlaty))

Omega, p = reinit_structs_cpu(fip.Omega, fip.p)
Lon, Lat = Omega.Lon, Omega.Lat
X, Y = Omega.X, Omega.Y

ufi3D_vec = Float64[]
ufi1D_vec = Float64[]
ulaty_vec = Float64[]

e1D_vec = [zeros(Omega.Nx * Omega.Ny) for _ in eachindex(tlaty)]
e3D_vec = [zeros(Omega.Nx * Omega.Ny) for _ in eachindex(tlaty)]

for k in eachindex(tlaty)
    k_fastiso = argmin( (t .- tlaty[k]./1e3) .^ 2 )
    ulatyitp = itp.(Lon, Lat, tlaty[k])
    ulaty_max[k] = maximum(abs.(ulatyitp))
    ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
    ufastiso1D = fip1D.out.u[k_fastiso] + fip1D.out.ue[k_fastiso]

    append!(ufi3D_vec, vec(ufastiso))
    append!(ufi1D_vec, vec(ufastiso1D))
    append!(ulaty_vec, vec(ulatyitp))

    e1D = abs.(ulatyitp - ufastiso1D)
    e3D = abs.(ulatyitp - ufastiso)
    e1D_vec[k] .= vec(e1D)
    e3D_vec[k] .= vec(e3D)

    mean_error[k] = mean(e3D)
    max_error[k] = maximum(e3D)
    mean_error1D[k] = mean(e1D)
    max_error1D[k] = maximum(e1D)
end

tlaty_vec = vcat([vec(fill(k, Omega.Nx, Omega.Ny)) for k in eachindex(tlaty)]...)
e1D_mat = vcat(e1D_vec...)
e3D_mat = vcat(e3D_vec...)

fig = Figure(resolution = (2150, 2300), fontsize = 40)
ax_uu = Axis(fig[1:3, 1:3], aspect = DataAspect())
ax_et = Axis(fig[1:3, 4:9], xticklabelrotation = π/2)
axbottom = [Axis(fig[i, j], aspect = DataAspect()) for j in [1:3, 4:6, 7:9],
    i in [4:6, 7:9]]
axs = vcat(ax_uu, ax_et, vec(axbottom))
umax = maximum(ulaty_max)

msmax = 3
ms = msmax .* abs.(ulaty_vec) ./ umax .+ 0.4
scatter!(axs[1], ulaty_vec, ufi1D_vec, markersize = ms, label = L"ELVA $\,$")
scatter!(axs[1], ulaty_vec, ufi3D_vec, markersize = ms, label = L"LV-ELVA $\,$")
lines!(axs[1], -600:50, -600:50, color = :gray10, label = label = L"identity $\,$")
axs[1].xlabel = L"$u_\mathrm{sk}$ (m)"
axs[1].ylabel = L"$u_\mathrm{fi}$ (m)"
axs[1].xticks = latexticks(-600:100:100)
axs[1].yticks = latexticks(-600:100:100)
axislegend(axs[1], position = :lt)
xlims!(axs[1], (-600, 70))
ylims!(axs[1], (-600, 70))
axislegend(axs[1], position = :rb)
axs[1].title = L"(a) $\,$"

bgap = 0.2
widthfactor = 2
boxplot!(axs[2], tlaty_vec .- bgap, e1D_mat ./ umax, width = widthfactor*bgap,
    label = L"ELVA $\,$")
boxplot!(axs[2], tlaty_vec .+ bgap, e3D_mat ./ umax, width = widthfactor*bgap,
    label = L"LV-ELVA $\,$")
axs[2].xlabel = L"Time (kyr) $\,$"
axs[2].ylabel = L"$e$ (1)"
axs[2].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
axs[2].yticks = latexticks(0:0.1:0.5)
axislegend(axs[2], position = :lt)
erellims = (0, 0.3)
ylims!(axs[2], erellims)
axs[2].yaxisposition = :right
axs[2].title = L"(b) $\,$"

ulims = (-500, 500)
u_opts = (colorrange = ulims, colormap = :PuOr)
e_opts = (colorrange = (-100, 100), colormap = :lighttemperaturemap)
k = argmax(max_error1D)
k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
ulatyitp = itp.(Lon, Lat, tlaty[k])
ufastiso = fip1D.out.u[k_fastiso] + fip1D.out.ue[k_fastiso]
hmu = heatmap!(axs[3], ulatyitp; u_opts...)
heatmap!(axs[4], X, Y, ufastiso; u_opts...)
hme = heatmap!(axs[5], ulatyitp - ufastiso; e_opts...)
axs[3].title = L"(c) Seakon, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[4].title = L"(d) ELVA, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[5].title = L"(e) (Seakon - ELVA), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

k = argmax(max_error)
k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
ulatyitp = itp.(Lon, Lat, tlaty[k])
ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
heatmap!(axs[6], ulatyitp; u_opts...)
heatmap!(axs[7], X, Y, ufastiso; u_opts...)
heatmap!(axs[8], ulatyitp - ufastiso; e_opts...)
axs[6].title = L"(f) Seakon, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[7].title = L"(g) LV-ELVA, $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[8].title = L"(h) (Seakon - LV-ELVA), $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"

[hidedecorations!(ax) for ax in axbottom]
Colorbar(fig[10, 2:5], hmu, label = L"Vertical displacement (m) $\,$", vertical = false,
    flipaxis = false, width = Relative(0.6), ticks = latexticks(-500:250:500))
Colorbar(fig[10, 7:9], hme, label = L"$u_\mathrm{sk} - u_\mathrm{fi} $ (m)",
    vertical = false, flipaxis = false, width = Relative(0.8), ticks = latexticks(-100:50:100))
save("plots/test4/$case-N=$N-final_v0.2.png", fig)
save("plots/test4/$case-N=$N-final_v0.2.pdf", fig)