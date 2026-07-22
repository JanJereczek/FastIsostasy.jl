using DSP
using FastIsostasy
using Interpolations
using LinearAlgebra
using Statistics
using Test

# The Enzyme-backed AD tests (rules, validity, adjoint, and AD-based inversions)
# dominate CI time with compilation overhead. Skip them with
# `FASTISOSTASY_TEST_AD=false julia test/runtests.jl` (or the equivalent Pkg.test
# preserve_env) when iterating on changes that cannot affect AD correctness; leave
# them on (the default) whenever Enzyme rules, custom pullbacks/pushforwards, or
# anything they differentiate through has changed.
const TEST_AD = lowercase(get(ENV, "FASTISOSTASY_TEST_AD", "true")) in ("1", "true", "yes")

include("test_aqua.jl")
include("test_jet.jl")
include("test_barystatic_sea_level.jl")
include("test_convolution.jl")
include("test_dataloaders.jl")
include("test_derivatives.jl")
include("test_dimensions.jl")
include("test_material.jl")
include("test_integrators.jl")
include("test_stability_diagnostics.jl")
include("test_simulated_observable.jl")
include("test_snapshot.jl")
include("test_progress.jl")
include("test_transient_creep.jl")
include("test_forward_recording.jl")
include("test_inversion_api.jl")

if TEST_AD
    include("test_ad_rules.jl")
    include("test_ad_validity.jl")
    include("test_adjoint_validity.jl")
    include("test_inversion_fullfield.jl")
    include("test_inversion_vialov.jl")
    include("test_inversion_viscdens.jl")
else
    @warn "FASTISOSTASY_TEST_AD is false: skipping Enzyme AD test files (rules, " *
        "validity, adjoint, AD-based inversions)."
end

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