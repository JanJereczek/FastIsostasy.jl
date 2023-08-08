push!(LOAD_PATH, "../")
using FastIsostasy
using Test, CairoMakie
using SpecialFunctions, Interpolations, LinearAlgebra
using JLD2, DelimitedFiles

include("helpers/compute.jl")
include("../publication_v1.0/test3/test3_cases.jl")
include("test_benchmarks.jl")
include("test_dimensions.jl")

@testset "FastIsostasy.jl" begin
    check_xy_ij()
    check_stereographic()
    benchmark1()
    benchmark2()
    benchmark3()
end