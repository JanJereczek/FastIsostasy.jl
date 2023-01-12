push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_compute.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    make_plot = true,
    make_anim = false,
)

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n, use_cuda = true)   # domain parameters
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)

    if occursin("2layers", case)
        eta_channel = fill(1e21, size(Omega.X)...)
        p = init_solidearth_params(T, Omega, channel_viscosity = eta_channel)
    elseif occursin("3layers", case)
        p = init_solidearth_params(T, Omega)
    end
    c = init_physical_constants(T)

    timespan = T.([0, 5e4]) * T(c.seconds_per_year)     # (yr) -> (s)
    dt_out = T(100) * T(c.seconds_per_year)             # (yr) -> (s), time step for saving output
    t_vec = timespan[1]:dt_out:timespan[2]              # (s)
    refine = fill(300.0, length(t_vec)-1)

    u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    # analytic_support = collect(10.0 .^ (-14:5))

    @testset "analytic solution" begin
        sol = analytic_solution(T(0), T(50000 * c.seconds_per_year), c, p, H, R, analytic_support)
        @test isapprox( sol, -1000*c.ice_density/mean(p.mantle_density), rtol=T(1e-2) )
    end

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    tools = precompute_terms(dt_out, Omega, p, c)

    @time forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz_disc, tools, p, c, dt_refine = refine)
    jldsave(
        "data/test1_$(case)_N=$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        Omega = Omega,
        c = c,
        p = p,
        R = R,
        H = H,
    )

end

"""
Application cases:
    - "cn_2layers"
    - "cn_3layers"
    - "euler_2layers"
    - "euler_3layers"
"""
case = "euler_2layers"
for n in 8:8
    main(n, case)
end