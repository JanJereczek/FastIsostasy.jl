push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
using Interpolations
include("helpers_compute.jl")
include("external_viscosity_maps.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    use_cuda = true::Bool,
)
    if use_cuda
        kernel = "gpu"
    else
        kernel = "cpu"
    end

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters

    c = init_physical_constants()

    log10_eta_channel = interpolate_visc_wiens_on_grid(Omega.X, Omega.Y)
    eta_channel = 10 .^ (log10_eta_channel)
    eta_halfspace = fill(10.0 ^ 21, size(Omega.X)...)
    p = init_solidearth_params(
        T,
        Omega,
        channel_viscosity = eta_channel,
        halfspace_viscosity = eta_halfspace,
    )

    lowest_eta = minimum(p.channel_viscosity[abs.(sigma_zz_disc) .> 1e-8])
    point_lowest_eta = argmin( (p.channel_viscosity .- lowest_eta).^2 )
    point_highest_eta = argmax(p.channel_viscosity .* abs.(sigma_zz_disc))
    points = [point_lowest_eta, point_highest_eta]

    timespan = T.([0, 1e4]) * T(c.seconds_per_year)     # (yr) -> (s)
    dt = T(10) * T(c.seconds_per_year)                  # (yr) -> (s)
    t_vec = timespan[1]:dt:timespan[2]                  # (s)

    u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)

    tools = precompute_terms(dt, Omega, p, c)

    @time forward_isostasy!(
        Omega,
        t_out,
        u3D_elastic,
        u3D_viscous,
        dudt3D_viscous,
        sigma_zz,
        tools,
        p,
        c,
    )
    jldsave(
        "data/discload_$(case)_N$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
    )

    ##############

    if make_anim
        anim_name = "plots/discload_$(case)_N=$(Omega.N)"
        animate_viscous_response(
            t_vec,
            Omega,
            u3D_viscous,
            anim_name,
            (-300.0, 50.0),
            points,
        )
    end
end

case = "wiens_viscosity_3layer"
main(7, case, make_anim = true)