push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
using CairoMakie
using Interpolations
include("helpers_compute.jl")
include("external_viscosity_maps.jl")

function main(n::Int, case::String; use_cuda::Bool = true, solver = "ExplicitEuler")

    T = Float64
    L = T(3000e3)                       # half-length of the square domain (m)
    Omega = ComputationDomain(L, n)     # domain parameters
    c = PhysicalConstants()
    if occursin("homogeneous", case)
        channel_viscosity = fill(1e20, Omega.N, Omega.N)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
        p = MultilayerEarth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    elseif occursin("meanviscosity", case)
        log10_eta_channel = interpolate_visc_wiens_on_grid(Omega.X, Omega.Y)
        channel_viscosity = 10 .^ (log10_eta_channel)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        lv = cat(channel_viscosity, halfspace_viscosity, dims=3)
        p = MultilayerEarth(
            Omega,
            c,
            layers_viscosity = lv,
        )
    elseif occursin("scaledviscosity", case)
        lb = [88e3, 180e3, 280e3, 400e3]
        halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)

        Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
        eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
            Omega.X, Omega.Y, Eta, Eta_mean)
        lv = 10.0 .^ cat(
            [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
            # [eta_interpolators[1].(Omega.X, Omega.Y) for itp in eta_interpolators]...,
            halfspace_logviscosity,
            dims=3,
        )
        p = MultilayerEarth(
            Omega,
            c,
            layers_begin = lb,
            layers_viscosity = lv,
        )
    end
    
    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.N) x $(Omega.N) grid...")

    R = T(2000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.(0.0:100:1e4)

    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, Hcylinder,
        active_geostate = false,
        ODEsolver=solver,
    )
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = copystructs2cpu(Omega, c, p)
    end

    jldsave(
        "data/test4/discload_$(case)_N$(Omega.N).jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        R = R, H = H,
    )

    ##############

end

cases = ["homogeneous_viscosity", "wiens_scaledviscosity", "wiens_meanviscosity"]
n = 8
for case in cases[1:2]
    main(n, case, use_cuda=false, solver=BS3())
end