push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
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
    println("Computing $case on $(Omega.N) x $(Omega.N) grid...")

    c = init_physical_constants()
    if occursin("2layers", case)
        layers_viscosity = [1e21, 1e21]
        p = init_multilayer_earth(Omega, c, layers_viscosity = layers_viscosity)
    elseif occursin("3layers", case)
        p = init_multilayer_earth(Omega, c)
    end

    timespan = years2seconds.([0.0, 5e4])           # (yr) -> (s)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    tools = precompute_fastiso(Omega, p, c)

    t1 = time()
    @time forward_isostasy!(
        Omega,
        t_out,
        u3D_elastic,
        u3D_viscous,
        dudt3D_viscous,
        sigma_zz_disc,
        tools,
        p,
        c,
    )
    t_fastiso = time() - t1

    if use_cuda
        Omega, p = copystructs2cpu(Omega, p, c)
    end

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
    )

end

"""
Application cases:
    - "cn2layers"
    - "cn3layers"
    - "euler2layers"
    - "euler3layers"
"""
case = "euler3layers"
for n in 4:6
    main(n, case, use_cuda = true)
end