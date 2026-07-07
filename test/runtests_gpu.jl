using FastIsostasy, Test
using CUDA

include("test_derivatives.jl")

@testset "gpu derivatives" begin
    domain, P, u, uxx, uyy, uxy = derivative_stdsetup(CuArray)
    test_derivatives(P, u, domain, uxx, uyy, uxy)
end