push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

function main(
    n::Int,             # 2^n cells on domain (1)
    case::String;       # Application case
    use_cuda = true::Bool,
)

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n, use_cuda = use_cuda)
    c = PhysicalConstants(rho_ice = 0.931e3, rho_litho = 3.3e3, rho_mantle = 3.3e3)
    #         layer_densities = [3.438e3, 3.871e3],

    G = 0.50605e11              # shear modulus (Pa)
    nu = 0.28
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]
    p = LateralVariability(
        Omega,
        layer_boundaries = lb,
        layer_viscosities = [1e21, 1e21, 2e21],
        litho_youngmodulus = E,
        litho_poissonratio = nu,
        litho_poissonratio = nu,
    )

    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.Nx) x $(Omega.Ny) grid...")

    if occursin("disc", case)
        alpha = T(10)                       # max latitude (°) of uniform ice disc
        Hmax = T(1000)                      # uniform ice thickness (m)
        R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
        H_ice = stereo_ice_cylinder(Omega, R, Hmax)
    elseif occursin("cap", case)
        alpha = T(10)                       # max latitude (°) of ice cap
        Hmax = T(1500)
        H_ice = stereo_ice_cap(Omega, alpha, Hmax)
    end
    t_out = years2seconds.([0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5])

    sl0 = fill(-Inf, Omega.Nx, Omega.Ny)
    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, H_ice, sealevel_0 = sl0,
        ODEsolver = "ExplicitEuler", interactive_geostate = true)
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = copystructs2cpu(Omega, c, p)
    end

    filename = "$(case)_Nx$(Omega.Nx)_Ny$(Omega.Ny)_$(kernel)"
    jldsave(
        "data/test2/$filename.jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        H = H_ice,
    )

end

cases = ["disc", "cap"]
for n in 6:6
    for case in cases
        main(n, case, use_cuda = false)
    end
end