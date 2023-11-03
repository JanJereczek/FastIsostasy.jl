push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, LinearAlgebra
include("cases.jl")
include("../../test/helpers/compute.jl")

function main(
    n::Int,                     # 2^n x 2^n cells on domain, (1)
    case::String;               # Application case
    use_cuda = false::Bool,
    dense_out = false::Bool,
)

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n, use_cuda = use_cuda)
    c = PhysicalConstants(rho_uppermantle = 3.6e3)
    p, _, _ = choose_case(case, Omega)
    
    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.Nx) x $(Omega.Ny) grid...")

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hice = uniform_ice_cylinder(Omega, R, H)
    if dense_out
        t_out = years2seconds.( vcat(0:1_000:5_000, 10_000:5_000:50_000) )
        densekey = "dense"
    else
        t_out = years2seconds.([0.0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
        densekey = "sparse"
    end

    fip = FastIsoProblem(Omega, c, p, t_out, false, Hice,
        diffeq = (alg = Tsit5(), reltol = 1e-5))
    solve!(fip)
    println("Took $(fip.out.computation_time) seconds!")
    println("-------------------------------------")

    filename = "$(case)_$(kernel)_Nx$(Omega.Nx)_Ny$(Omega.Ny)_$(densekey)_premparams"
    @save "../data/test3/$filename.jld2" fip Hice
end

for n in 7:7
    # ["gaussian_lo_D", "gaussian_hi_D", "no_litho", "ref",
    #     "gaussian_lo_η", "gaussian_hi_η", "homogeneous"]
    for case in ["gaussian_lo_η", "no_litho", "ref"]
        main(n, case, use_cuda = false, dense_out = true)
    end
end