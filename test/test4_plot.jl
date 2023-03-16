using CairoMakie


if occursin("homogeneous", case) | occursin("meanviscosity", case)
    checkfig = Figure(resolution = (1600, 700), fontsize = 20)
    labels = [
        L"$z \in [88, 400]$ km",
        L"$z \, > \, 400$ km",
        L"Equivalent half-space viscosity $\,$",
    ]
elseif occursin("scaledviscosity", case)
    checkfig = Figure(resolution = (1600, 550), fontsize = 20)
    labels = [
        L"$z \in [88, 180]$ km",
        L"$z \in ]180, 280]$ km",
        L"$z \in ]280, 400]$ km",
        L"$z \, > \, 400$ km",
        L"Equivalent half-space viscosity $\,$",
    ]
end

clim = (18.0, 23.0)
cmap = cgrad(:jet, rev = true)
for k in axes(lv, 3)
    ax = Axis(
        checkfig[1, k],
        aspect = DataAspect(),
        title = labels[k],
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        ylabel = L"$y \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
    )
    if k > 1
        hideydecorations!(ax)
    end
    heatmap!(
        ax,
        Omega.X,
        Omega.Y,
        log10.(lv[:, :, k])',
        colormap = cmap,
        colorrange = clim,
    )
    scatter!(
        [Omega.X[20, 24], Omega.X[36, 38]],
        [Omega.Y[20, 24], Omega.Y[36, 38]],
        color = :white,
        markersize = 20,
    )
end
ax = Axis(
    checkfig[1, size(lv, 3)+1],
    aspect = DataAspect(),
    title = labels[size(lv, 3)+1],
    xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
    ylabel = L"$y \: \mathrm{(10^3 \: km)}$",
    xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
    yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
)
hideydecorations!(ax)

hm = heatmap!(
    ax,
    Omega.X,
    Omega.Y,
    log10.(p.effective_viscosity)',
    colormap = cmap,
    colorrange = clim,
)
scatter!(
    [Omega.X[20, 24], Omega.X[36, 38]],
    [Omega.Y[20, 24], Omega.Y[36, 38]],
    color = :white,
    markersize = 20,
)
Colorbar(
    checkfig[2, :],
    hm,
    vertical = false,
    width = Relative(0.3),
    label = L"log viscosity $\,$",
)
save("plots/test4/$(case)_visclayers.png", checkfig)

if make_anim
    anim_name = "plots/test4/discload_$(case)_N$(Omega.N)"
    animate_viscous_response(
        t_out,
        Omega,
        u3D_viscous,
        anim_name,
        (-300.0, 50.0),
        points,
        20,
    )
end