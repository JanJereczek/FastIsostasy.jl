push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using JLD2
include("helpers_computation.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
)

    T = Float64
    L = T(2000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters
    p = init_solidearth_params(
        T,
        Omega,
        channel_viscosity = fill(T(1e21), Omega.N, Omega.N),
        halfspace_viscosity = fill(T(1e22), Omega.N, Omega.N),
        channel_begin = fill(T(70e3), Omega.N, Omega.N),
        halfspace_begin = fill(T(670e3), Omega.N, Omega.N),
    )
    c = init_physical_constants(T, ice_density = 0.931e3)

    timespan = years2seconds.(T.([0, 1e5]))     # (yr) -> (s)
    dt_out = years2seconds(T(100))              # (yr) -> (s), time step for saving output
    t_vec = timespan[1]:dt_out:timespan[2]      # (s)

    u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)

    if case == "disc"
        alpha = T(10)                       # max latitude (°) of uniform ice disc
        H = T(1000)                         # uniform ice thickness (m)
        R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
        sigma_zz = generate_uniform_disc_load(Omega, c, R, H)
    elseif case == "cap"
        alpha = T(10)                       # max latitude (°) of ice cap
        H = T(1500)
        sigma_zz = generate_cap_load(Omega, c, alpha, H)
    end

    tools = precompute_terms(dt_out, Omega, p, c)
    @time forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz, tools, p, c)
    jldsave(
        "data/test2_$(case)_N$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        sigma_zz = sigma_zz,
        Omega = Omega,
        c = c,
        p = p,
        t_vec = t_vec,
    )
end

cases = ["disc", "cap"]
for n in 7:7
    for case in cases
        N = 2^n
        println("Computing $case on $N x $N grid...")
        main(n, case)
    end
end