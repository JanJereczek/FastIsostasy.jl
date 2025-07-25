using FastIsostasy
using Test, CairoMakie, Statistics

include("helpers/benchmark_constants.jl")
include("helpers/compute.jl")
include("helpers/plot.jl")
include("../publication_v1.0/test3/cases.jl")
include("../publication_v1.0/helpers_computation.jl")
include("test_derivatives.jl")

init()
const SAVE_PLOTS = true

@testset "gpu derivatives" begin
    domain, P, u, uxx, uyy, uxy = derivative_stdsetup(true)
    test_derivatives(P, u, domain, uxx, uyy, uxy)
end

@testset "FastIsostasy.jl" begin
    benchmark1_gpu()
end

