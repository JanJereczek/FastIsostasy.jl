push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using Interpolations
include("helpers_plot.jl")
include("helpers_compute.jl")
include("external_viscosity_maps.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    make_anim = false,
)

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters
    R = T(1500e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    c = init_physical_constants()
    if occursin("meanviscosity", case)
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

    checkfig = Figure(resolution = (1600, 900))
    clim = (18.0, 23.0)
    cmap = :jet
    for k in axes(lv, 3)
        ax = Axis(checkfig[1, k], aspect = DataAspect())
        heatmap!(ax, log10.(lv[:, :, k]), colormap = cmap, colorrange = clim)
    end
    ax = Axis(checkfig[1, size(lv, 3)+1], aspect = DataAspect())
    hm = heatmap!(ax, log10.(p.effective_viscosity), colormap = cmap, colorrange = clim)
    Colorbar(checkfig[2, :], hm, vertical = false, width = Relative(0.3))
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

    lowest_eta = minimum(p.effective_viscosity[Omega.X.^2 + Omega.Y.^2 .< (1.3e6)^2])
    point_lowest_eta = argmin( (p.effective_viscosity .- lowest_eta).^2 )
    point_highest_eta = argmax(p.effective_viscosity .* abs.(sigma_zz_disc))
    points = [point_lowest_eta, point_highest_eta]

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

cases = ["wiens_scaledviscosity", "wiens_meanviscosity"]
for case in cases
    main(6, case, make_anim = true)
end