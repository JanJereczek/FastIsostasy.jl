using FastIsostasy, Test

include("test_derivatives.jl")

@testset "gpu derivatives" begin
    domain, P, u, uxx, uyy, uxy = derivative_stdsetup(true)
    test_derivatives(P, u, domain, uxx, uyy, uxy)
end