push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers_compute.jl")

function main(
    n::Int,                     # 2^n x 2^n cells on domain, (1)
    case::String;               # Application case
)

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(L, n)
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    c = PhysicalConstants()
    if occursin("2layers", case)
        layers_viscosity = [1e21, 1e21]
        p = MultilayerEarth(Omega, c, layers_viscosity = layers_viscosity)
    elseif occursin("3layers", case)
        p = MultilayerEarth(Omega, c)
    end

    timespan = years2seconds.([0.0, 5e4])           # (yr) -> (s)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)

    @testset "analytic solution" begin
        sol = analytic_solution(
            T(0), timespan[end], c, p, H, R, analytic_support)
        @test isapprox( sol, -1000*c.ice_density/mean(p.mean_density), rtol=T(1e-2) )
    end

end

for case in ["2layers", "3layers"]
    for n in 6:8
        main(n, case)
    end
end