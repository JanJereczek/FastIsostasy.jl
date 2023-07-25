push!(LOAD_PATH, "../")
using FastIsostasy
using Test
using SpecialFunctions
using JLD2, DelimitedFiles
using Interpolations
include("autotest1.jl")
include("helpers_compute.jl")

@testset "FastIsostasy.jl" begin
    check_xy_ij()
    check_stereographic()
    # benchmark1()
    benchmark2()
end
