using CairoMakie
using DelimitedFiles
using DSP
using FastIsostasy
using Interpolations
using LinearAlgebra
using Statistics
using Test

include("helpers/benchmark_constants.jl")
include("helpers/compute.jl")
include("helpers/plot.jl")
include("../publication_v1.0/test3/cases.jl")
include("../publication_v1.0/helpers_computation.jl")

const SAVE_PLOTS = false

include("test_adaptive_ocean.jl")
include("test_convolution.jl")
include("test_dataloaders.jl")
include("test_derivatives.jl")
include("test_elra.jl")
include("test_dimensions.jl")

include("test_benchmarks.jl")
@testset "benchmarks" begin
    benchmark1()
    benchmark1_external_loadupdate()
    benchmark2()
    benchmark3()
    # benchmark5()
end