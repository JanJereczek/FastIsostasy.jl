using DSP
using FastIsostasy
using Interpolations
using LinearAlgebra
using Statistics
using Test

include("test_aqua.jl")
include("test_jet.jl")
include("test_barystatic_sea_level.jl")
include("test_convolution.jl")
include("test_dataloaders.jl")
include("test_derivatives.jl")
include("test_dimensions.jl")
include("test_integrators.jl")
include("test_simulated_observable.jl")
include("test_snapshot.jl")
include("test_forward_recording.jl")
include("test_inversion_api.jl")
include("test_ad_rules.jl")
include("test_ad_validity.jl")
include("test_adjoint_validity.jl")
include("test_inversion_fullfield.jl")
include("test_inversion_vialov.jl")
include("test_inversion_viscdens.jl")

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