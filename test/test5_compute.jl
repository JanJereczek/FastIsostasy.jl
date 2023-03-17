push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")
include("external_load_maps.jl")
include("external_viscosity_maps.jl")

function main(n::Int, active_gs::Bool; use_cuda::Bool = false,solver = "ExplicitEuler")

    T = Float64
    L = T(4000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(L, n)   # domain parameters
    c = PhysicalConstants()

    lb = [88e3, 180e3, 280e3, 400e3]
    halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)
    Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    p = MultilayerEarth(
        Omega,
        c,
        layers_begin = lb,
        layers_viscosity = lv,
    )

    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.N) x $(Omega.N) grid...")

    t_out, deltaH, H = interpolated_glac1d_snapshots(Omega)
    dH = [deltaH[:, :, k] for k in axes(deltaH, 3)]
    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, t_out, dH,
        active_geostate = active_gs,
        ODEsolver=solver,
        dt = years2seconds(0.1),
    )
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = copystructs2cpu(Omega, c, p)
    end

    case = active_gs ? "geostate" : "isostate"
    jldsave(
        "data/test5/$(case)_N$(Omega.N).jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        H = dH,
    )
end

cases = [false, true]
for active_gs in cases[1:1]
    main(6, active_gs, use_cuda=true, solver="ExplicitEuler")
end