push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using CairoMakie
include("helpers_plot.jl")
include("external_load_maps.jl")
include("external_viscosity_maps.jl")

function animate_deglaciation(
    t_vec::AbstractVector{T},
    Omega::ComputationDomain,
    u::Array{T, 3},
    dudt::Array{T, 3},
    H3D::Array{T, 3},
    H3D_anom::Array{T, 3},
    anim_name::String,
    u_range::Tuple{T, T},
    framerate::Int,
) where {T<:AbstractFloat}

    t_vec = seconds2years.(t_vec)
    i = Observable(1)
    u2D = @lift(u[:, :, $i]')
    dudt2D = @lift(m_per_sec2mm_per_yr.(dudt[:, :, $i]'))
    load2D = @lift(H3D[:, :, $i]')
    load2D_anom = @lift(H3D_anom[:, :, $i]')

    fig = Figure(resolution = (1600, 650))
    ax1 = Axis(
        fig[2, 1],
        aspect = DataAspect(),
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        ylabel = L"$y \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
    )
    ax2 = Axis(
        fig[2, 2],
        aspect = DataAspect(),
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticklabelsvisible = false,
    )
    ax3 = Axis(
        fig[2, 3],
        aspect = DataAspect(),
        xlabel = L"$x \: \mathrm{(10^3 \: km)}$",
        xticks = (-3e6:1e6:3e6, num2latexstring.(-3:3)),
        yticklabelsvisible = false,
    )

    cmapload = cgrad(:balance)
    cmapdudt = cgrad([:grey20, :bisque, :orange, :red3, :turquoise2])
    # cmapu = cgrad(:cool, rev = true)
    cmapu = cgrad([:magenta, :lavenderblush, :cyan, :cornflowerblue, :royalblue1])
    lims_load = (-1e3, 1e3)
    lims_dudt = (-3.5, 10.5)
    lims_u = u_range

    hm_load = heatmap!(
        ax1,
        Omega.X,
        Omega.Y,
        load2D_anom,
        colorrange = lims_load,
        colormap = cmapload,
    )
    contour!(
        ax1,
        Omega.X[1,:],
        Omega.Y[:,1],
        load2D,
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
        load2D,
        linewidth = 1,
        color = :black,
    )
    Colorbar(
        fig[1, 2],
        hm_dudt,
        label = L"Uplift rate $\dot{u}^V$ (mm/year)",
        vertical = false,
        width = Relative(0.8),
    )

    hm_u = heatmap!(
        ax3,
        Omega.X,
        Omega.Y,
        u2D,
        colorrange = lims_u,
        colormap = cmapu,
        highclip = :royalblue3
    )
    contour!(
        ax3,
        Omega.X[1,:],
        Omega.Y[:,1],
        load2D,
        linewidth = 1,
        color = :black,
    )
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
    #     load2D,
    #     levels = 5,
    #     transformation = (:xy, 0.0),
    #     color = :white,
    #     transparency = true,
    # )

    Colorbar(
        fig[1, 3],
        hm_u,
        label = L"Viscous displacement $u^V$ (m)",
        vertical = false,
        width = Relative(0.8),
    )

    record(fig, "$anim_name.mp4", axes(u, 3), framerate = framerate) do k
            i[] = k
    end
end


function main(
    n::Int,             # 2^n cells on domain (1)
    case::String,       # Application case
)

    N = 2^n
    sol = load("data/test5/$(case)_N$(N).jld2")
    anim_name = "plots/test5/loaduplift_$(case)_N$(N)"
    t_out, deltaH, H = interpolated_glac1d_snapshots(sol["Omega"])
    load_itp = linear_interpolation(t_out, [H[:, :, k] for k in axes(H,3)])
    load_out = cat( [ load_itp(t) for t in sol["t_out"] ]..., dims = 3 )
    anom_itp = linear_interpolation(t_out, [deltaH[:, :, k] for k in axes(deltaH,3)])
    load_anom = cat( [ anom_itp(t) for t in sol["t_out"] ]..., dims = 3 )
    animate_deglaciation(
        sol["t_out"],
        sol["Omega"],
        sol["u3D_viscous"],
        sol["dudt3D_viscous"],
        load_out,
        load_anom,
        anim_name,
        (-100.0, 300.0),
        24,
    )
end

cases = ["glac1dload", "ice7gload"]
for case in cases[1:1]
    main(7, case)
end