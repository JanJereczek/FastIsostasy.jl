push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using Interpolations
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

    log10_eta_channel = interpolate_visc_wiens_on_grid(Omega.X, Omega.Y)
    channel_viscosity = 10 .^ (log10_eta_channel)
    halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
    lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
    p = init_multilayer_earth(
        Omega,
        c,
        layers_viscosity = lv,
    )

    t_out_yr = 0.0:10:1e4
    t_out = years2seconds.(t_out_yr)
    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)
    tools = precompute_fastiso(Omega, p, c)
    dt = fill( years2seconds(1.0), length(t_out)-1 )

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
    )

    ##############

    lowest_eta = minimum(channel_viscosity[abs.(sigma_zz_disc) .> 1e-8])
    point_lowest_eta = argmin( (channel_viscosity .- lowest_eta).^2 )
    point_highest_eta = argmax(channel_viscosity .* abs.(sigma_zz_disc))
    points = [point_lowest_eta, point_highest_eta]

    if make_anim
        anim_name = "plots/discload_$(case)_N=$(Omega.N)"
        animate_viscous_response(
            t_out,
            Omega,
            u3D_viscous,
            anim_name,
            (-300.0, 50.0),
            points,
        )
    end
end

case = "wiens_viscosity_3layer"
main(6, case, make_anim = true)