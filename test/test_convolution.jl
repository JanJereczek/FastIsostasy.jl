@testset "in-place convolutions" begin
    N = 300
    A, B = rand(N, N), rand(N, N)
    ipconv = InplaceConvolution(A, false)
    Cipc = ipconv(B)
    C = conv(A, B)
    @test Cipc ≈ C
end