push!(LOAD_PATH, "../")
using FastIsostasy
using Test, CairoMakie
using Interpolations, LinearAlgebra, Statistics
using JLD2, DelimitedFiles, DSP

include("helpers/compute.jl")
include("../publication_v1.0/test3/cases.jl")
include("../publication_v1.0/helpers.jl")
include("test_benchmarks.jl")
include("test_dimensions.jl")
include("test_derivatives.jl")
include("convolution.jl")

const SAVE_PLOTS = false
@testset "FastIsostasy.jl" begin
    check_xy_ij()
    # check_stereographic()
    # check_derivatives()
    benchmark1()
    benchmark1_external_loadupdate()
    benchmark2()
    benchmark3()
    # benchmark5()
    test_convolution()
end