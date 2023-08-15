push!(LOAD_PATH, "../")
using FastIsostasy
using Test, CairoMakie, Statistics

include("helpers/compute.jl")
include("test_benchmarks.jl")
include("test_derivatives.jl")

init()
const make_plots = true

@testset "FastIsostasy.jl" begin
    check_gpu_derivatives()
    benchmark1_gpu()
end