push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using CairoMakie
using Interpolations
include("helpers_plot.jl")
include("helpers_compute.jl")
include("external_viscosity_maps.jl")

function main(
    case::String;       # Application case
    make_anim = false,
)

    # Fix resolution --> fast but not 100% accurate results for sanity check.
    n = 6
    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters
    R = T(2000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    c = init_physical_constants()
    if occursin("homogeneous", case)
        channel_viscosity = fill(1e20, Omega.N, Omega.N)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
        p = init_multilayer_earth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    elseif occursin("meanviscosity", case)
        log10_eta_channel = interpolate_visc_wiens_on_grid(Omega.X, Omega.Y)
        channel_viscosity = 10 .^ (log10_eta_channel)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
        p = init_multilayer_earth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    elseif occursin("scaledviscosity", case)
        lb = [88e3, 180e3, 280e3, 400e3]
        halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)

        Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
        eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
            Omega.X, Omega.Y, Eta, Eta_mean)
        lv = 10.0 .^ cat(
            [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
            # [eta_interpolators[1].(Omega.X, Omega.Y) for itp in eta_interpolators]...,
            halfspace_logviscosity,
            dims=3,
        )
        p = init_multilayer_earth(
            Omega,
            c,
            layers_begin = lb,
            layers_viscosity = lv,
        )
    end

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
    save("plots/test4a/$(case)_visclayers.png", checkfig)

    t_out_yr = 0.0:100:2e4
    t_out = years2seconds.(t_out_yr)
    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)
    tools = precompute_fastiso(Omega, p, c)
    if n >= 7
        dt = fill( years2seconds(0.1), length(t_out)-1 )
    else
        dt = fill( years2seconds(1.0), length(t_out)-1 )
    end

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    sigma_zz_snapshots = ([t_out[1], t_out[end]], [sigma_zz_disc, sigma_zz_disc])

    t1 = time()
    @time forward_isostasy!(
        Omega,
        t_out,
        u3D_elastic,
        u3D_viscous,
        dudt3D_viscous,
        sigma_zz_snapshots,
        tools,
        p,
        c,
        dt = dt,
    )
    t_fastiso = time() - t1

    # if use_cuda
    #     Omega, p = copystructs2cpu(Omega, p, c)
    # end

    # lowest_eta = minimum(p.effective_viscosity[Omega.X.^2 + Omega.Y.^2 .< (1.8e6)^2])
    # point_lowest_eta = argmin( (p.effective_viscosity .- lowest_eta).^2 )
    # highest_eta = maximum(p.effective_viscosity[Omega.X.^2 + Omega.Y.^2 .< (1.8e6)^2]) 
    # point_highest_eta = argmin( (p.effective_viscosity .- highest_eta).^2 )
    # points = [point_lowest_eta, point_highest_eta]
    # display(points) 
    points = [CartesianIndex(20, 24), CartesianIndex(36, 38)]

    jldsave(
        "data/test4a/discload_$(case)_N$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        dudt3D_viscous = dudt3D_viscous,
        sigma_zz = sigma_zz_disc,
        Omega = Omega,
        c = c,
        p = p,
        R = R,
        H = H,
        t_fastiso = t_fastiso,
        t_out = t_out,
        eta_extrema = points,
    )

    ##############
    if make_anim
        anim_name = "plots/test4a/discload_$(case)_N$(Omega.N)"
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
end

cases = ["homogeneous_viscosity", "wiens_scaledviscosity", "wiens_meanviscosity"]
for case in cases
    main(case, make_anim = true)
end