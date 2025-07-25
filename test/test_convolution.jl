@testset "in-place convolutions" begin
    T = Float32
    N = 64
    kernel, input = [rand(T, N, N) for _ in 1:2]
    helpers = ConvolutionPlanHelpers(kernel)
    convplan = ConvolutionPlan(kernel, helpers)
    FastIsostasy.conv!(input, convplan, helpers)
    
    output_dsp = DSP.conv(kernel, input)
    @test output_dsp == helpers.output_cropped
end