using FastIsostasy, Test
using CUDA

const TEST_AD = lowercase(get(ENV, "FASTISOSTASY_TEST_AD", "true")) in ("1", "true", "yes")

include("test_derivatives.jl")

@testset "gpu derivatives" begin
    domain, P, u, uxx, uyy, uxy = derivative_stdsetup(CuArray)
    test_derivatives(P, u, domain, uxx, uyy, uxy)
end

include("test_integrators_gpu.jl")

# Enzyme through KernelAbstractions kernels on CUDA. Requires the CUDA AD rules in
# `ext/FastIsostasyEnzymeCUDAExt.jl`, which load once `Enzyme` and `CUDA` are both
# present. Slow to compile (see the note in the file). Skip with
# `FASTISOSTASY_TEST_AD=false` when iterating on non-AD changes.
if TEST_AD
    include("test_ad_validity_gpu.jl")
else
    @warn "FASTISOSTASY_TEST_AD is false: skipping GPU Enzyme AD validity tests."
end