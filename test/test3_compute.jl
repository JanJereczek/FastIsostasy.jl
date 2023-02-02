push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("test3_cases.jl")
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
    p = choose_case(case, Omega, c)
    t_out_yr = [0.0, 1.0, 1e1, 1e2, 1e3, 2e3, 5e3, 1e4, 1e5]
    t_out = years2seconds.(t_out_yr)
    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    tools = precompute_fastiso(Omega, p, c)
    dt = fill( years2seconds(1.0), length(t_out)-1 )

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
        dt = dt,
    )
    t_fastiso = time() - t1

    if use_cuda
        Omega, p = copystructs2cpu(Omega, p, c)
    end

    jldsave(
        "data/test3/$filename.jld2",
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
end

#= Application cases:
["binaryD", "binaryη", "binaryDη"]
["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η"]
=#
for n in 5:5
    for case in ["gaussian_hi_η"]
        main(n, case, use_cuda = false)
    end
end