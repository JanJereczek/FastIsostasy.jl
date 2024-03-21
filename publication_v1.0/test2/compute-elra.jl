using FastIsostasy
using JLD2
include("../../test/helpers/compute.jl")

function main(n::Int, case::String)

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n)
    # c = PhysicalConstants(rho_ice = 0.931e3, rho_uppermantle = 3.3e3)
    c = PhysicalConstants(rho_ice = 0.931e3, rho_litho = 2.8e3)
    # layer_densities = [3.438e3, 3.871e3],
    b = fill(1e6, Omega.Nx, Omega.Ny)   # elevated bedrock to prevent any load from ocean

    p = LayeredEarth(Omega, tau = years2seconds(5e3))

    println("Computing on $(Omega.Nx) x $(Omega.Ny) grid...")
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

    εt = 1e-8
    t_out = years2seconds.([-εt, 0.0, 1.0, 1e3, 2e3, 5e3, 1e4, 1e5])
    t_Hice = [-εt, 0.0, t_out[end]]
    Hice = [zeros(Omega.Nx, Omega.Ny), H_ice, H_ice]
    mask = collect(H_ice .> 1e-8)

    opts = SolverOptions(interactive_sealevel = true, verbose = true,
        diffeq = (alg=Tsit5(), reltol=1e-4), deformation_model = :elra)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice,
        opts = opts, b_0 = b, maskactive = mask)
    solve!(fip)
    println("Computation took $(fip.out.computation_time) seconds!")
    println("-------------------------------------")

    filename = "elra-$(case)_Nx=$(Omega.Nx)_Ny=$(Omega.Ny)"
    path = "../data/test2/$filename"
    @save "$path.jld2" fip
    savefip("$path.nc", fip)
end

cases = ["disc", "cap"]
for n in 7:7
    for case in cases
        main(n, case)
    end
end



# using CairoMakie
# using FastIsostasy

# n = 6
# use_cuda = false
# dense = false

# T = Float64
# W = T(3000e3)               # half-length of the square domain (m)
# Omega = ComputationDomain(W, n, use_cuda = use_cuda, correct_distortion = false)
# c = PhysicalConstants(rho_litho = 0.0)
# p = LayeredEarth(Omega, layer_viscosities = [1e21], layer_boundaries = [88e3])

# R = T(1000e3)               # ice disc radius (m)
# H = T(1000)                 # ice disc thickness (m)
# Hcylinder = uniform_ice_cylinder(Omega, R, H)

# L_w = get_flexural_lengthscale(mean(p.litho_rigidity), c.rho_uppermantle, c.g)
# kei = get_kei(Omega, L_w)
# viscousgreen = calc_viscous_green(Omega, p, kei, L_w)
# viscousconv = InplaceConvolution(viscousgreen, Omega.use_cuda)
# u_equil = viscousconv(-Hcylinder * c.g * c.rho_ice)
# heatmap(u_equil)
# heatmap(Hcylinder)