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

Hlim = (1e-8, 4e3)
kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
axs = [Axis(fig[1, j], title = @lift("t = $(t[$kobs]) ka"),
    aspect = DataAspect()) for j in 1:2]
[hidedecorations!(ax) for ax in axs]
hm1 = heatmap!(axs[1], @lift(Hice_vec[$kobs]), colorrange = Hlim, lowclip = :transparent,
    colormap = :ice)
hm2 = heatmap!(axs[2], @lift(fip.out.u[$kobs]), colormap = :vik, colorrange = (-800, 800))
Colorbar(fig[2, 1], hm1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], hm2, vertical = false, flipaxis = false, width = Relative(0.8))
if make_anims
    record(fig, "plots/test4/$case-N=$N-cycle-displacement.mp4",
        eachindex(t), framerate = 10) do k
        kobs[] = k
    end
end

dHlims = (-2600, 2600)
ulims = (-500, 500)
kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
vars = [deltaH, fip.out.u .+ fip.out.ue]
axs = [Axis3(fig[1, j], title = @lift("t = $(t[$kobs]) ka,"*
 "  range = $(round.(extrema(vars[j][$kobs])))")) for j in 1:2]
# [hidedecorations!(ax) for ax in axs]
sf1 = surface!(axs[1], @lift(deltaH[$kobs]), colorrange = dHlims, colormap = :vik)
sf2 = surface!(axs[2], @lift(fip.out.u[$kobs] + fip.out.ue[$kobs]), colormap = :PuOr,
    colorrange = ulims)
zlims!(axs[1], dHlims)
zlims!(axs[2], ulims)
Colorbar(fig[2, 1], sf1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], sf2, vertical = false, flipaxis = false, width = Relative(0.8))
if make_anims
    record(fig, "plots/test4/$case-N=$N-cycle-surface.mp4",
        eachindex(t), framerate = 10) do k
        kobs[] = k
    end
end

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

    # tmpfig = Figure(resolution = (1800, 700), fontsize = 30)
    # axs = [Axis(tmpfig[1, j]) for j in 1:3]
    # [hidedecorations!(ax) for ax in axs]
    # hm1 = heatmap!(axs[1], ulatyitp, colorrange = ulims, colormap = :PuOr)
    # hm2 = heatmap!(axs[2], ufastiso, colorrange = ulims, colormap = :PuOr)
    # hm3 = heatmap!(axs[3], ulatyitp - ufastiso, colorrange = elims,
    #     colormap = :lighttemperaturemap)
    # Colorbar(tmpfig[2, 1:2], hm1, label = "vertical displacement (m)", vertical = false,
    #     width = Relative(0.4))
    # Colorbar(tmpfig[2, 3], hm3, label = L"$u_\mathrm{sk} - u_\mathrm{fi} $ (m)",
    #     vertical = false, width = Relative(0.8))
    # save("plots/test4/displacements/$case-N=$N-t=$(tlaty[k]).png", tmpfig)
end


fig = Figure(resolution = (2150, 1300), fontsize = 34)
axtop = Axis(fig[1:2, 2:5], xticklabelrotation = π/2)
# axfill = [Axis(fig[1:2, 1]), Axis(fig[1:2, 6])]
# hidedecorations!.(axfill)
# hidespines!.(axfill)
axbottom = [Axis(fig[3:6, j], aspect = DataAspect()) for j in [1:2, 3:4, 5:6]]
axs = vcat(axtop, axbottom)
umax = maximum(ulaty_max)
bwidth = 0.35
barplot!(axs[1], eachindex(tlaty) .- 0.2, max_error ./ umax,
    width = bwidth, label = L"max LV-ELVA $\,$")
barplot!(axs[1], eachindex(tlaty) .- 0.2, mean_error ./ umax,
    width = bwidth, label = L"mean LV-ELVA $\,$")
barplot!(axs[1], eachindex(tlaty) .+ 0.2, max_error1D ./ umax,
    width = bwidth, label = L"max ELVA $\,$")
barplot!(axs[1], eachindex(tlaty) .+ 0.2, mean_error1D ./ umax,
    width = bwidth, label = L"mean ELVA $\,$")

axs[1].xlabel = L"Time (kyr) $\,$"
axs[1].ylabel = L"$e$ (1)"
axs[1].xticks = (eachindex(tlaty), latexify(round.(tlaty ./ 1e3, digits = 3)))
axs[1].yticks = latexticks(0:0.1:0.5)
axislegend(axs[1], position = :lt)

erellims = (0, 0.3)
ylims!(axs[1], erellims)

k = argmax(max_error)
k_fastiso = argmin( (t .- tlaty[k]/1e3) .^ 2 )
ulatyitp = itp.(Lon, Lat, tlaty[k])
ufastiso = fip.out.u[k_fastiso] + fip.out.ue[k_fastiso]
[hidedecorations!(ax) for ax in axbottom]
hm1 = heatmap!(axs[2], ulatyitp, colorrange = ulims, colormap = :PuOr)
hm2 = heatmap!(axs[3], X, Y, ufastiso, colorrange = ulims, colormap = :PuOr)
hm3 = heatmap!(axs[4], ulatyitp - ufastiso, colorrange = (-100, 100),
    colormap = :lighttemperaturemap)
# arc!(axs[3], Point2f(-700e3, -400e3), 200e3, -π, π)

Colorbar(fig[7, 2:3], hm1, label = L"Vertical displacement (m) $\,$", vertical = false,
    flipaxis = false, width = Relative(0.8), ticks = latexticks(-500:250:500))
Colorbar(fig[7, 5:6], hm3, label = L"$u_\mathrm{sk} - u_\mathrm{fi} $ (m)",
    vertical = false, flipaxis = false, width = Relative(0.8), ticks = latexticks(-100:50:100))
axs[2].title = L"Seakon at $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[3].title = L"FastIsostasy at $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
axs[4].title = L"Difference at $t = %$(Int(round(tlaty[k] ./ 1e3)))$ kyr"
save("plots/test4/$case-N=$N-final.png", fig)
save("plots/test4/$case-N=$N-final.pdf", fig)

msmax = 3
ms = msmax .* abs.(ulaty_vec) ./ umax .+ 0.4
fig, ax, sc = scatter(ulaty_vec, ufi1D_vec, markersize = ms)
scatter!(ulaty_vec, ufi3D_vec, color = :orange, markersize = ms)
lines!(ax, -600:50, -600:50, color = :gray10)
fig

tlaty_vec = vcat([vec(fill(k, Omega.Nx, Omega.Ny)) for k in eachindex(tlaty)]...)
e1D_mat = vcat(e1D_vec...)
e3D_mat = vcat(e3D_vec...)

bgap = 0.2
widthfactor = 2
fig, ax, bp = boxplot(tlaty_vec .- bgap, e1D_mat, width = widthfactor*bgap)
boxplot!(ax, tlaty_vec .+ bgap, e3D_mat, width = widthfactor*bgap)
fig

#=
viscfig = Figure(resolution = (2000, 1000), fontsize = 30)
axs = [Axis(viscfig[1, j], aspect = DataAspect()) for j in 1:2]
[hidedecorations!(ax) for ax in axs]

hm1 = heatmap!(axs[1], fip.Omega.X, fip.Omega.Y, fip.p.effective_viscosity,
    colormap = cgrad(:jet, rev=true))
arc!(axs[1], Point2f(-700e3, -400e3), 200e3, -π, π, color = :black, linewidth = 5)
Colorbar(viscfig[2, 1], hm1, vertical = false, width = Relative(0.8))

hm2 = heatmap!(axs[2], fip.Omega.X, fip.Omega.Y, fip.p.litho_thickness,
    colormap = cgrad(:inferno, rev=true))
arc!(axs[2], Point2f(-700e3, -400e3), 200e3, -π, π, color = :black, linewidth = 5)
Colorbar(viscfig[2, 2], hm2, vertical = false, width = Relative(0.8))

viscfig

loadfig = Figure(resolution = (2000, 1000))
axs = [Axis(loadfig[1, j], aspect = DataAspect()) for j in 1:2]
[hidedecorations!(ax) for ax in axs]
opts = (colormap = :ice, colorrange = (1e-8, 2000), lowclip = :white)
Colorbar(loadfig[2, :], vertical = false, width = Relative(0.5); opts...)
for k in eachindex(t)
    heatmap!(axs[1], fip1D.out.Hice[k]; opts...)
    heatmap!(axs[2], fip1D.out.Hice[k]; opts...)
    save("plots/test4/loads/$k.png", loadfig)
end
=#