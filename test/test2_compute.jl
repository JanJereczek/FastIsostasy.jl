push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using JLD2
include("helpers_compute.jl")

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
    Omega = init_domain(L, n, use_cuda = use_cuda)
    filename = "$(case)_$(kernel)_N$(Omega.N)"

    G = 0.50605e11              # shear modulus (Pa)
    nu = 0.5
    E = G * 2 * (1 + nu)
    c = init_physical_constants(ice_density = 0.931e3)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]

    p = init_multilayer_earth(
        Omega,
        c,
        layers_begin = lb,
        layers_density = [3.438e3, 3.871e3],
        # layers_density = [3.438e3, 3.871e3],
        layers_viscosity = [1e21, 1e21, 2e21],
        litho_youngmodulus = E,
        litho_poissonratio = nu,
    )

    t_out_yr = [0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5]
    t_out = years2seconds.(t_out_yr)

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)

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
    sigma_zz_snapshots = ([t_out[1], t_out[end]], [sigma_zz, sigma_zz])
    tools = precompute_fastiso(Omega, p, c)

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
    )
    t_fastiso = time() - t1

    if use_cuda
        Omega, p = copystructs2cpu(Omega, p, c)
    end
    jldsave(
        "data/test2/$filename.jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        dudt3D_viscous = dudt3D_viscous,
        sigma_zz = sigma_zz,
        Omega = Omega,
        c = c,
        p = p,
        t_fastiso = t_fastiso,
        t_out = t_out,
    )

end

cases = ["disc", "cap"]
for n in 6:8
    for case in cases
        N = 2^n
        println("Computing $case on $N x $N grid...")
        main(n, case, use_cuda = true)
    end
end