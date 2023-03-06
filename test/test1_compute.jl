push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

function main(
    n::Int;                     # 2^n x 2^n cells on domain, (1)
    use_cuda::Bool = false,
    solver::Any = "SimpleEuler",
)
    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n, use_cuda = use_cuda)
    c = init_physical_constants()
    p = init_multilayer_earth(Omega, c)

    kernel = use_cuda ? "gpu" : "cpu"
    filename = "$(solver)_N$(Omega.N)_$kernel"
    println("Computing on $kernel and $(Omega.N) x $(Omega.N) grid...")

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_Hice_snapshots = [t_out[1], t_out[end]]
    Hice_snapshots = [Hcylinder, Hcylinder]

    t_eta_snapshots = [t_out[1], t_out[end]]
    eta_snapshots = kernelpromote([p.effective_viscosity, p.effective_viscosity], Array)

    tools = precompute_fastiso(Omega, p, c)
    t1 = time()
    results = forward_isostasy(t_out, Omega, tools, p, c,
        t_Hice_snapshots, Hice_snapshots, t_eta_snapshots, eta_snapshots,
        ODEsolver = solver)
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = copystructs2cpu(Omega, p, c)
    end

    jldsave(
        "data/test1/$filename.jld2",
        Omega = Omega,
        c = c,
        p = p,
        results = results,
        t_fastiso = t_fastiso,
        R = R,
        H = H,
    )
end

# for s in ["SimpleEuler", BS3(), Tsit5()]
for s in ["SimpleEuler"]
    for n in 4:8
        main(n, use_cuda = true, solver = s)
    end
end