push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using JLD2
include("helpers_compute.jl")

@inline function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
)

    T = Float64
    L = T(2200e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n)   # domain parameters
    
    litho_thickness = 70e3      # lithosphere thickness (m)
    G = 0.50605e11              # shear modulus (Pa)
    nu = 0.5
    E = G * 2 * (1 + nu)
    D = (E * litho_thickness ^ 3)/(12*(1-nu^2))

    p = init_solidearth_params(
        T,
        Omega,
        lithosphere_rigidity = fill(T(D), Omega.N, Omega.N),
        mantle_density = fill(T(3.6e3), Omega.N, Omega.N),
        channel_viscosity = fill(T(1e21), Omega.N, Omega.N),
        halfspace_viscosity = fill(T(2e21), Omega.N, Omega.N),
        channel_begin = fill(T(litho_thickness), Omega.N, Omega.N),
        halfspace_begin = fill(T(670e3), Omega.N, Omega.N),
    )
    c = init_physical_constants(T, ice_density = 0.931e3)

    t_out_yr = [0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5]
    t_out = years2seconds.(t_out_yr)
    refine = diff(t_out_yr)

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)

    if occursin("disc", case)
        alpha = T(10)                       # max latitude (°) of uniform ice disc
        H = T(1000)                         # uniform ice thickness (m)
        R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
        sigma_zz = generate_uniform_disc_load(Omega, c, R, H)
    elseif occursin("cap", case)
        alpha = T(10)                       # max latitude (°) of ice cap
        H = T(1500)
        sigma_zz = generate_cap_load(Omega, c, alpha, H)
    end

    tools = precompute_terms(1.0, Omega, p, c)
    @time forward_isostasy!(Omega, t_out, u3D_elastic, u3D_viscous, sigma_zz, tools, p, c, dt_refine = refine)

    # @time forward_isostasy!(Omega, t_out, u3D_elastic, u3D_viscous, sigma_zz, tools, p, c, viscous_solver = "CrankNicolson")

    jldsave(
        "data/test2_$(case)_N$(Omega.N).jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        sigma_zz = sigma_zz,
        Omega = Omega,
        c = c,
        p = p,
        t_vec = t_out,
    )
end

cases = ["disc", "cap"]
for n in 6:6
    for case in cases
        N = 2^n
        println("Computing $case on $N x $N grid...")
        main(n, case)
    end
end