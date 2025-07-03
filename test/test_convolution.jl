@testset "in-place convolutions" begin
    N = 300
    A, B = rand(N, N), rand(N, N)
    ipconv = InplaceConvolution(A, false)
    convolution!(ipconv, B)
    C = conv(A, B)
    @test ipconv.out ≈ C
end

T = Float32
N = 64
kernel, input = [rand(T, N, N) for _ in 1:2]
convplan = ConvolutionPlan(kernel)
FastIsostasy.conv!(input, convplan)

output_dsp = DSP.conv(kernel, input)
output_dsp == convplan.output_cropped
