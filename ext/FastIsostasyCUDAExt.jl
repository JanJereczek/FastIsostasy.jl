module FastIsostasyCUDAExt

using CUDA
import FastIsostasy

CUDA.allowscalar(false)

FastIsostasy.cudainfo() = CUDA.versioninfo()

# CUDA FFT plans — overrides the CPU default in tools.jl. Out-of-place (applied
# via `mul!`) to preserve inputs, matching the CPU path.
FastIsostasy.choose_fft_plans(X::CuArray) = (
    CUDA.CUFFT.plan_fft(complex.(X)),
    CUDA.CUFFT.plan_ifft(complex.(X))
)

# The spatial-derivative and thin-plate kernels are backend-generic (launched via
# `get_backend` in derivatives.jl / deformation.jl), so no CUDA-specific
# overrides are needed for them anymore.

end
