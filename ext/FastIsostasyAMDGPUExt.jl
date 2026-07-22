module FastIsostasyAMDGPUExt

# AMD GPU support. This file is deliberately the whole of it — if adding a vendor
# ever needs more than `allowscalar(false)` plus a `deviceinfo()`, the backend
# abstraction in src/utils.jl (`kernelzeros`/`kernelpromote`) or the FFT planner in
# src/tools.jl (`_planner_flags`) has sprung a leak and should be fixed there
# rather than patched here. See the comment in `ext/FastIsostasyCUDAExt.jl`.
#
# Usage:
#     using FastIsostasy, AMDGPU
#     domain = RegionalDomain(3000e3, 7, backend = ROCBackend())

using AMDGPU
import FastIsostasy

AMDGPU.allowscalar(false)

# Local method, looked up by `FastIsostasy.deviceinfo` via `Base.get_extension`.
deviceinfo() = AMDGPU.versioninfo()

end
