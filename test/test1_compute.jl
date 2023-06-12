push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

function main(
    n::Int;                     # 2^n x 2^n cells on domain, (1)
    use_cuda::Bool = false,
    solver::Any = "ExplicitEuler",
    active_gs::Bool = true,
)
    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n, use_cuda = use_cuda)
    c = PhysicalConstants()
    p = MultilayerEarth(Omega, c)

    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.N) x $(Omega.N) grid...")

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=solver, interactive_geostate=active_gs)
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = copystructs2cpu(Omega, c, p)
    end

    gs = active_gs ? "geostate" : "isostate"
    filename = "$(solver)_N$(Omega.N)_$(kernel)_$(gs)"
    jldsave(
        "data/test1/$filename.jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        R = R, H = H,
    )
end

# ["ExplicitEuler", BS3(), VCABM(), Rosenbrock23(autodiff=false)]
for use_cuda in [true] # [false, true]
    for active_gs in [false] # [false, true]
        for n in 4:6 # 3:8
            main(n, use_cuda = use_cuda, solver = "ExplicitEuler", active_gs = active_gs)
        end
    end
end
