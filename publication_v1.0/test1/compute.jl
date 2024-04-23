using FastIsostasy
using JLD2
include("../../test/helpers/compute.jl")

function main(n::Int; use_cuda::Bool = false, dense::Bool = false)
    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n, use_cuda = use_cuda, correct_distortion = false)
    c = PhysicalConstants(rho_litho = 0.0)
    p = LayeredEarth(Omega, layer_viscosities = [1e21], layer_boundaries = [88e3])

    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.Ny) x $(Omega.Nx) grid...")
    filename = "Nx=$(Omega.Nx)_Ny=$(Omega.Ny)_$(kernel)"
    if dense
        filename *= "-dense"
        t_out = years2seconds.(0.0:100:50_000)
    else
        t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    end

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]
    
    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice)
    solve!(fip)
    println("Computation took $(fip.out.computation_time) seconds!")
    println("-------------------------------------")

    path = "$(@__DIR__)/../../data/test1/$filename"
    @save "$path.jld2" fip
    savefip("$path.nc", fip)
end

for use_cuda in [true]
    for n in 3:8
        main(n, use_cuda = use_cuda, dense = false)
    end
end

#=
Slight speed up if using powers of 2:

This file:
main(n, use_cuda = false, solver = BS3(), active_gs = false)
Took 0.6100420951843262 seconds!

main(n, use_cuda = false, solver = "SimpleEuler", active_gs = false)
Took 14.107969999313354 seconds!

------------------------------------

test1_rectangle.jl:
main(63, 64, use_cuda = false, solver = BS3(), active_gs = false)
Took 0.6303250789642334 seconds!

main(63, 64, use_cuda = false, solver = "SimpleEuler", active_gs = false)
Took 14.486158847808838 seconds!
=#