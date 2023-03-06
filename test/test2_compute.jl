push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    use_cuda = true::Bool,
    solver::Any = "SimpleEuler",
)

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n, use_cuda = use_cuda)
    c = init_physical_constants(ice_density = 0.931e3)

    G = 0.50605e11              # shear modulus (Pa)
    nu = 0.5
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]
    p = init_multilayer_earth(
        Omega,
        c,
        layers_begin = lb,
        layers_density = [3.438e3, 3.871e3],
        layers_viscosity = [1e21, 1e21, 2e21],
        litho_youngmodulus = E,
        litho_poissonratio = nu,
    )

    kernel = use_cuda ? "gpu" : "cpu"
    filename = "$(case)_N$(Omega.N)_$(kernel)"

    t_out = years2seconds.([0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5])
    if occursin("disc", case)
        alpha = T(10)                       # max latitude (°) of uniform ice disc
        Hmax = T(1000)                      # uniform ice thickness (m)
        R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
        H_ice = uniform_ice_cylinder(Omega, R, Hmax)
    elseif occursin("cap", case)
        alpha = T(10)                       # max latitude (°) of ice cap
        Hmax = T(1500)
        H_ice = ice_cap(Omega, c, alpha, Hmax)
    end

    t_Hice_snapshots = [t_out[1], t_out[end]]
    Hice_snapshots = [H_ice, H_ice]

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
        "data/test2/$filename.jld2",
        Omega = Omega,
        c = c,
        p = p,
        results = results,
        t_fastiso = t_fastiso,
    )

end

cases = ["disc", "cap"]
for n in 8:8
    for case in cases
        N = 2^n
        println("Computing $case on $N x $N grid...")
        main(n, case, use_cuda = true)
    end
end