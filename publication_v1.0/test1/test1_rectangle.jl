push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("../test/helpers/compute.jl")

function main(
    Nx::Int,
    Ny::Int;
    use_cuda::Bool = false,
    solver::Any = "ExplicitEuler",
    active_gs::Bool = true,
)
    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, W, Nx, Ny, use_cuda = use_cuda)
    c = PhysicalConstants()
    p = LateralVariability(Omega)

    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.Ny) x $(Omega.Nx) grid...")

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=solver,
        interactive_geostate=active_gs)
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = reinit_structs_cpu(Omega, p)
    end

    gs = active_gs ? "geostate" : "isostate"
    filename = "$(solver)_Nx$(Omega.Nx)_Ny$(Omega.Ny)_$(kernel)_$(gs)"
    jldsave(
        "../data/test1/$filename.jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        R = R, H = H,
    )
end

main(63, 64, use_cuda = false, solver = "ExplicitEuler", active_gs = false)

