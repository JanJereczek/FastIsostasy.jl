using FastIsostasy
using JLD2, LinearAlgebra
include("cases.jl")
include("../../test/helpers/compute.jl")

dir = @__DIR__

function main(n::Int)

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n)
    c = PhysicalConstants(rho_uppermantle = 3.6e3)
    p = LayeredEarth(Omega, tau = years2seconds(3e3), layer_boundaries = [100e3, 600e3])
    t_out = years2seconds.( vcat(0.0:1_000:5_000, 10_000:5_000:50_000) )

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    Hice = [zeros(Omega.nx, Omega.ny), Hcylinder, Hcylinder]

    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]

    opts = SolverOptions(verbose = true, deformation_model = :elra)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts)
    solve!(fip)
    println("Took $(fip.out.computation_time) seconds!")
    println("-------------------------------------")

    filename = "elra_Nx$(Omega.nx)_ny$(Omega.ny)"
    path = "$dir/../../data/test3/$filename"
    @save "$path.jld2" fip
    savefip("$path.nc", fip)
end

main(7)