module FastIsostasyCUDAExt

using CUDA
import FastIsostasy

CUDA.allowscalar(false)

# Local method (looked up by `FastIsostasy.cudainfo` via `Base.get_extension`).
# Defining it here rather than overwriting `FastIsostasy.cudainfo()` avoids the
# "method overwriting during precompilation" error.
cudainfo() = CUDA.versioninfo()

# CUDA FFT plans — overrides the CPU default in tools.jl. Out-of-place (applied
# via `mul!`) to preserve inputs, matching the CPU path. The inverse plan is wrapped
# by `normalize_plan` so its normalization is Enzyme-inactive (see convolutions.jl).
FastIsostasy.choose_fft_plans(X::CuArray) = (
    CUDA.CUFFT.plan_fft(complex.(X)),
    FastIsostasy.normalize_plan(CUDA.CUFFT.plan_ifft(complex.(X)))
)

# The spatial-derivative and thin-plate kernels are backend-generic (launched via
# `get_backend` in derivatives.jl / deformation.jl), so no CUDA-specific
# overrides are needed for them anymore.

end
