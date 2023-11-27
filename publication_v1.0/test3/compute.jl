push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, LinearAlgebra
include("cases.jl")
include("../../test/helpers/compute.jl")

function main(n::Int, case::String; dense_out = false::Bool)

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n)
    c = PhysicalConstants(rho_uppermantle = 3.6e3)
    p, _, _ = choose_case(case, Omega)
    tol = occursin("_D", case) ? 1e-5 : 1e-4

    if dense_out
        t_out = years2seconds.( vcat(0.0:1_000:5_000, 10_000:5_000:50_000) )
        densekey = "dense"
    else
        t_out = years2seconds.([0.0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
        densekey = "sparse"
    end

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]

    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]

    opts = SolverOptions(diffeq = (alg = Tsit5(), reltol = tol), verbose = true)
    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts)
    solve!(fip)
    println("Took $(fip.out.computation_time) seconds!")
    println("-------------------------------------")

    filename = "$(case)_Nx$(Omega.Nx)_Ny$(Omega.Ny)_$(densekey)"
    path = "../data/test3/$filename"
    @save "$path.jld2" fip
    savefip("$path.nc", fip)
end

cases = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η", "no_litho",
    "ref", "homogeneous"]
for n in 7:7
    for case in cases[7:7]
        main(n, case, dense_out = true)
    end
end