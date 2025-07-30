using DSP
using FastIsostasy
using Interpolations
using LinearAlgebra
using Statistics
using Test

include("test_barystatic_sea_level.jl")
include("test_convolution.jl")
include("test_dataloaders.jl")
include("test_derivatives.jl")
include("test_dimensions.jl")

# const SAVE_PLOTS = true

# include("helpers/benchmark_constants.jl")
# include("helpers/compute.jl")
# include("helpers/plot.jl")
# include("helpers/cases.jl")
# include("../publication_v1.0/helpers_computation.jl")

# include("test_benchmarks.jl")
# @testset "benchmarks" begin
#     benchmark1()
#     benchmark1_float32()
#     benchmark1_external_loadupdate()
#     benchmark1_gpu()
#     benchmark2()
#     benchmark3()
#     benchmark5()
# end