push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("test3_cases.jl")
include("helpers_compute.jl")

function main(
    n::Int,                     # 2^n x 2^n cells on domain, (1)
    case::String;               # Application case
    use_cuda = false::Bool,
    dense_out = false::Bool,
)

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n, use_cuda = use_cuda)
    c = PhysicalConstants()
    p = choose_case(case, Omega, c)
    
    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.Nx) x $(Omega.Ny) grid...")

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    if dense_out
        t_out = years2seconds.( vcat(0:1_000:5_000, 10_000:5_000:50_000) )
        densekey = "dense"
    else
        t_out = years2seconds.([0.0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
        densekey = "sparse"
    end

    t1 = time()
    results = fastisostasy(
        t_out, Omega, c, p, Hcylinder,
        interactive_geostate=false,
        ODEsolver="ExplicitEuler")
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = copystructs2cpu(Omega, c, p)
    end

    filename = "$(case)_$(kernel)_Nx$(Omega.Nx)_Ny$(Omega.Ny)_$densekey"
    jldsave(
        "data/test3/$filename.jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        R = R, H = H,
    )
end

for n in 7:7
    # ["gaussian_lo_D", "gaussian_hi_D", , "gaussian_no_D", "gaussian_lo_η", "gaussian_hi_η"]
    for case in ["gaussian_lo_D", "gaussian_hi_D"]
        main(n, case, use_cuda = false, dense_out = true)
    end
end