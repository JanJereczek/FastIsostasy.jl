push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_compute.jl")

@inline function main(
    n::Int,                     # 2^n x 2^n cells on domain, (1)
    case::String;               # Application case
    use_cuda = true::Bool,
)

    if use_cuda
        kernel = "gpu"
    else
        kernel = "cpu"
    end

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n, use_cuda = use_cuda)
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    filename = "$(case)_$(kernel)_N$(Omega.N)"

    if occursin("2layers", case)
        eta_channel = fill(1e21, size(Omega.X)...)
        p = init_solidearth_params(T, Omega, channel_viscosity = eta_channel)
    elseif occursin("3layers", case)
        p = init_solidearth_params(T, Omega)
    end
    c = init_physical_constants(T)

    timespan = years2seconds.([0.0, 5e4])           # (yr) -> (s)
    dt_out = years2seconds(100.0)                   # (yr) -> (s), time step for saving output
    t_out = timespan[1]:dt_out:timespan[2]          # (s)
    refine = fill(100.0, length(t_out)-1)

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
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

    t1 = time()
    @time forward_isostasy!(Omega, t_out, u3D_elastic, u3D_viscous, sigma_zz_disc, tools, p, c, dt_refine = refine)
    t_fastiso = time() - t1
    Omega, p = copystructs2cpu(Omega, p)

    jldsave(
        "data/test1/$filename.jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        Omega = Omega,
        c = c,
        p = p,
        R = R,
        H = H,
        t_fastiso = t_fastiso,
        t_out = t_out,
        refine = refine,
    )

end

"""
Application cases:
    - "cn2layers"
    - "cn3layers"
    - "euler2layers"
    - "euler3layers"
"""
case = "euler2layers"
for n in 5:5
    main(n, case, use_cuda = true)
end