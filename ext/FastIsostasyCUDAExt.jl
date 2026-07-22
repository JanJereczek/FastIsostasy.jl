module FastIsostasyCUDAExt

using CUDA
import FastIsostasy

CUDA.allowscalar(false)

# Local method (looked up by `FastIsostasy.deviceinfo` via `Base.get_extension`).
# Defining it here rather than overwriting `FastIsostasy.deviceinfo()` avoids the
# "method overwriting during precompilation" error.
deviceinfo() = CUDA.versioninfo()

# That is the whole extension. Nothing else here is CUDA-specific:
#
#  • Arrays are allocated through `KernelAbstractions.allocate` off the domain's
#    `backend` (`kernelzeros`/`kernelpromote` in src/utils.jl), so no `CuArray`
#    constructor is ever named.
#  • The spatial-derivative and thin-plate kernels are backend-generic, launched
#    via `get_backend` (derivatives.jl / deformation.jl).
#  • FFT plans no longer need a CUDA method: `plan_fft`/`plan_ifft` are
#    `AbstractFFTs` generics that CUDA.jl already claims for `CuArray`, and the one
#    non-portable piece — FFTW's `flags = MEASURE` — is now selected off the
#    backend in `_planner_flags` (src/tools.jl).
#
# See `ext/FastIsostasyAMDGPUExt.jl`, which is the same file with the vendor
# swapped, for the check that this really is all a backend costs.

end
