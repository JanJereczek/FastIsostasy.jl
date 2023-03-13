push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using CairoMakie
using Interpolations
include("helpers_plot.jl")
include("helpers_compute.jl")
include("external_load_maps.jl")
include("external_viscosity_maps.jl")

function main(
    n::Int,             # 2^n cells on domain (1)
    case::String,       # Application case
)

    T = Float64
    L = T(4000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(L, n)   # domain parameters
    c = PhysicalConstants()

    lb = [88e3, 180e3, 280e3, 400e3]
    halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)
    Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    p = MultilayerEarth(
        Omega,
        c,
        layers_begin = lb,
        layers_viscosity = lv,
    )

    t_out, deltaH, H = interpolated_glac1d_snapshots(Omega)
    delta_sigma = - (c.g * c.rho_ice) .* deltaH
    sigma_zz_snapshots = (t_out, [delta_sigma[:, :, k] for k in axes(delta_sigma, 3) ])

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)
    tools = PrecomputedFastiso(Omega, p, c)
    if n >= 7
        dt = fill( years2seconds(0.1), length(t_out)-1 )
    else
        dt = fill( years2seconds(1.0), length(t_out)-1 )
    end

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
    #     Omega, p = copystructs2cpu(Omega, c, p)
    # end

    lowest_eta = minimum(p.effective_viscosity[abs.(deltaH[:, :, end]) .> 1])
    point_lowest_eta = argmin( (p.effective_viscosity .- lowest_eta).^2 )
    point_highest_eta = argmax(p.effective_viscosity .* abs.(deltaH))
    points = [point_lowest_eta, point_highest_eta]

    jldsave(
        "data/test4b/$(case)_N$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        dudt3D_viscous = dudt3D_viscous,
        sigma_zz_snapshots = sigma_zz_snapshots,
        Omega = Omega,
        c = c,
        p = p,
        t_fastiso = t_fastiso,
        t_out = t_out,
        eta_extrema = points,
    )
end

cases = ["glac1dload", "ice7gload"]
for case in cases[1:1]
    main(7, case)
end