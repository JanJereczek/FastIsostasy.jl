using FastIsostasy, Test
using CUDA

include("test_derivatives.jl")

@testset "gpu derivatives" begin
    domain, P, u, uxx, uyy, uxy = derivative_stdsetup(CuArray)
    test_derivatives(P, u, domain, uxx, uyy, uxy)
end

# Enzyme through KernelAbstractions kernels on CUDA. Requires the CUDA AD rules in
# `ext/FastIsostasyEnzymeCUDAExt.jl`, which load once `Enzyme` and `CUDA` are both
# present. Slow to compile (see the note in the file).
include("test_ad_validity_gpu.jl")