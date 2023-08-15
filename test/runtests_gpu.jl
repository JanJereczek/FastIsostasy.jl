push!(LOAD_PATH, "../")
using FastIsostasy
using Test, CairoMakie

include("helpers/compute.jl")
include("test_benchmarks.jl")
include("test_derivatives.jl")

init()
@testset "FastIsostasy.jl" begin
    check_gpu_derivatives()
    benchmark1_gpu()
end