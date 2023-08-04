push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using CairoMakie
using LaTeXStrings
include("../test/helpers/plot.jl")
include("../test/helpers/loadmaps.jl")
include("../test/helpers/viscmaps.jl")

function main(
    n::Int,             # 2^n cells on domain (1)
    case::String,       # Application case
)

    N = 2^n
    filekey = "$(case)_Nx$(N)_Ny$(N)"
    sol = load("../data/test5/$filekey.jld2")
    results = sol["results"]
    t_out = results.t_out
    t_out_kyr = round.(seconds2years.(t_out) ./ 1e3, digits=1)
    H_anom, Omega = sol["H"], sol["Omega"]
    u = results.viscous
    dudt = results.displacement_rate

    # We also want to have the absolute load
    _, __, H = interpolated_glac1d_snapshots(Omega)

    i = Observable(1)
    u2D = @lift(u[$i]')
    dudt2D = @lift(m_per_sec2mm_per_yr.(dudt[$i]'))
    H2D_anom = @lift(H_anom[$i]')
    H2D = @lift(H[:, :, $i]')
    time_stamp = @lift(latexstring( string("t = ", t_out_kyr[$i], L"\: \mathrm{kyr}")))

    fig = Figure(resolution = (1500, 650), fontsize=22)
    ax1 = Axis(
        fig[2, 1],
        aspect = DataAspect(),
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        ylabel = L"$y \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        title = L"$\textbf{(a)}$",
    )
    ax2 = Axis(
        fig[2, 2],
        aspect = DataAspect(),
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticklabelsvisible = false,
        title = L"$\textbf{(b)}$",
        #title = time_stamp,
    )
    ax3 = Axis(
        fig[2, 3],
        aspect = DataAspect(),
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticklabelsvisible = false,
        title = L"$\textbf{(c)}$",
    )
    xlims!(ax1, (-3e6, 3e6))
    xlims!(ax2, (-3e6, 3e6))
    xlims!(ax3, (-3e6, 3e6))
    ylims!(ax1, (-3e6, 3e6))
    ylims!(ax2, (-3e6, 3e6))
    ylims!(ax3, (-3e6, 3e6))

    # cmapload = cgrad(:RdBu)
    cmapload = cgrad([:red4, :firebrick1, :lightsalmon, :white, :cornflowerblue])
    cmapdudt = cgrad([
        :grey20,
        :grey60,
        :honeydew1,
        :papayawhip,
        :khaki1,
        :sandybrown,
        :firebrick1,
        :magenta,
        :midnightblue,
        :turquoise1,
        :turquoise2,
    ])
    # cmapdudt = cgrad([
    #     :grey20,
    #     :bisque,
    #     :orange,
    #     :red3,
    #     :midnightblue,
    #     :turquoise2,
    # ])
    cmapu = cgrad([:magenta, :white, :cyan, :cornflowerblue, :royalblue1, :royalblue3, :midnightblue])
    lims_load = (-1.5e3, 0.5e3)
    lims_dudt = (-3.5, 10.5)
    lims_u = (-100.0, 500.0)
    Hticks = (-1500:500:500, num2latexstring.(-1500:500:500))
    dudtticks = (-4:2:10, num2latexstring.(-4:2:10))
    uticks = (-100:100:500, num2latexstring.(-100:100:500))

    hm_load = heatmap!(
        ax1,
        Omega.X,
        Omega.Y,
        H2D_anom,
        colorrange = lims_load,
        colormap = cmapload,
    )
    contour!(
        ax1,
        Omega.X[1,:],
        Omega.Y[:,1],
        H2D,
        # levels = 0.0:1.0,
        linewidth = 1,
        color = :black,
    )
    Colorbar(
        fig[1, 1],
        hm_load,
        label = L"Thickness anomaly $\Delta H$ (m)",
        vertical = false,
        width = Relative(0.8),
        ticks = Hticks,
        minorticks = IntervalsBetween(2),
        minorticksvisible = true,
    )

    hm_dudt = heatmap!(
        ax2,
        Omega.X,
        Omega.Y,
        dudt2D,
        colorrange = lims_dudt,
        colormap = cmapdudt,
    )
    contour!(
        ax2,
        Omega.X[1,:],
        Omega.Y[:,1],
        H2D,
        linewidth = 1,
        color = :black,
    )
    Colorbar(
        fig[1, 2],
        hm_dudt,
        label = L"Displacement rate $\dot{u}$ (mm/year)",
        vertical = false,
        width = Relative(0.8),
        ticks = dudtticks,
    )

    hm_u = heatmap!(
        ax3,
        Omega.X,
        Omega.Y,
        u2D,
        colorrange = lims_u,
        colormap = cmapu,
        # highclip = :royalblue3
    )
    contour!(
        ax3,
        Omega.X[1,:],
        Omega.Y[:,1],
        H2D,
        linewidth = 1,
        color = :black,
    )
    
    Colorbar(
        fig[1, 3],
        hm_u,
        label = L"Vertical displacement $u$ (m)",
        vertical = false,
        width = Relative(0.8),
        ticks = uticks,
    )

    framerate = 24
    plotname = "plots/test5/loaduplift_$filekey"
    record(fig, "$plotname.mp4", eachindex(u), framerate = framerate) do k
            i[] = k
    end

    save("$plotname.png", fig)
    save("$plotname.pdf", fig)
end

cases = ["isostate", "geostate"]
for case in cases[1:1]
    main(6, case)
end


# function animate_deglaciation(
#     t_vec::AbstractVector{T},
#     Omega::ComputationDomain,
#     u::Array{T, 3},
#     dudt::Array{T, 3},
#     H3D::Array{T, 3},
#     H3D_anom::Array{T, 3},
#     anim_name::String,
#     lims_u::Tuple{T, T},
#     framerate::Int,
# ) where {T<:AbstractFloat}

#     t_vec = seconds2years.(t_vec)
#     i = Observable(1)
#     u2D = @lift(u[:, :, $i]')
#     dudt2D = @lift(m_per_sec2mm_per_yr.(dudt[:, :, $i]'))
#     load2D = @lift(H3D[:, :, $i]')
#     load2D_anom = @lift(H3D_anom[:, :, $i]')

#     fig = Figure(resolution = (1600, 650))
#     ax1 = Axis(
#         fig[2, 1],
#         aspect = DataAspect(),
#         xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
#         ylabel = L"$y \: \mathrm{(10^3 \: km)}$",
#         xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
#         yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
#     )
#     ax2 = Axis(
#         fig[2, 2],
#         aspect = DataAspect(),
#         xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
#         xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
#         yticklabelsvisible = false,
#     )
#     ax3 = Axis(
#         fig[2, 3],
#         aspect = DataAspect(),
#         xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
#         xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
#         yticklabelsvisible = false,
#     )

#     cmapload = cgrad(:balance)
#     cmapdudt = cgrad([:grey20, :bisque, :orange, :red3, :turquoise2])
#     # cmapu = cgrad(:cool, rev = true)
#     cmapu = cgrad([:magenta, :lavenderblush, :cyan, :cornflowerblue, :royalblue1])
#     lims_load = (-1e3, 1e3)
#     lims_dudt = (-3.5, 10.5)
#     lims_u = lims_u

#     hm_load = heatmap!(
#         ax1,
#         Omega.X,
#         Omega.Y,
#         load2D_anom,
#         colorrange = lims_load,
#         colormap = cmapload,
#     )
#     contour!(
#         ax1,
#         Omega.X[1,:],
#         Omega.Y[:,1],
#         load2D,
#         # levels = 0.0:1.0,
#         linewidth = 1,
#         color = :black,
#     )
#     Colorbar(
#         fig[1, 1],
#         hm_load,
#         label = L"Thickness anomaly $\Delta H$ (m)",
#         vertical = false,
#         width = Relative(0.8),
#     )

#     hm_dudt = heatmap!(
#         ax2,
#         Omega.X,
#         Omega.Y,
#         dudt2D,
#         colorrange = lims_dudt,
#         colormap = cmapdudt,
#     )
#     contour!(
#         ax2,
#         Omega.X[1,:],
#         Omega.Y[:,1],
#         load2D,
#         linewidth = 1,
#         color = :black,
#     )
#     Colorbar(
#         fig[1, 2],
#         hm_dudt,
#         label = L"Uplift rate $\dot{u}^V$ (mm/year)",
#         vertical = false,
#         width = Relative(0.8),
#     )

#     hm_u = heatmap!(
#         ax3,
#         Omega.X,
#         Omega.Y,
#         u2D,
#         colorrange = lims_u,
#         colormap = cmapu,
#         highclip = :royalblue3
#     )
#     contour!(
#         ax3,
#         Omega.X[1,:],
#         Omega.Y[:,1],
#         load2D,
#         linewidth = 1,
#         color = :black,
#     )
#     # sf = surface!(
#     #     ax3,
#     #     1e-6 .* Omega.X,
#     #     1e-6 .* Omega.Y,
#     #     u2D,
#     #     colorrange = lims_u,
#     #     colormap = cmapu,
#     # )
#     # wireframe!(
#     #     ax3,
#     #     1e-6 .* Omega.X,
#     #     1e-6 .* Omega.Y,
#     #     u2D,
#     #     linewidth = 0.08,
#     #     color = :black,
#     # )
#     # contour!(
#     #     ax3,
#     #     1e-6 .* Omega.X[1,:],
#     #     1e-6 .* Omega.Y[:,1],
#     #     load2D,
#     #     levels = 5,
#     #     transformation = (:xy, 0.0),
#     #     color = :white,
#     #     transparency = true,
#     # )

#     Colorbar(
#         fig[1, 3],
#         hm_u,
#         label = L"Viscous displacement $u^V$ (m)",
#         vertical = false,
#         width = Relative(0.8),
#     )

#     record(fig, "$anim_name.mp4", axes(u, 3), framerate = framerate) do k
#             i[] = k
#     end
# end

# animate_deglaciation(
#     sol["t_out"],
#     sol["Omega"],
#     sol["u3D_viscous"],
#     sol["dudt3D_viscous"],
#     load_out,
#     load_anom,
#     anim_name,
#     (-100.0, 300.0),
#     24,
# )

# sf = surface!(
    #     ax3,
    #     1e-6 .* Omega.X,
    #     1e-6 .* Omega.Y,
    #     u2D,
    #     colorrange = lims_u,
    #     colormap = cmapu,
    # )
    # wireframe!(
    #     ax3,
    #     1e-6 .* Omega.X,
    #     1e-6 .* Omega.Y,
    #     u2D,
    #     linewidth = 0.08,
    #     color = :black,
    # )
    # contour!(
    #     ax3,
    #     1e-6 .* Omega.X[1,:],
    #     1e-6 .* Omega.Y[:,1],
    #     H2D,
    #     levels = 5,
    #     transformation = (:xy, 0.0),
    #     color = :white,
    #     transparency = true,
    # )