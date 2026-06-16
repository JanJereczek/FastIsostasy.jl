using FastIsostasy
using CUDA
using Chairmarks
using KernelAbstractions: get_backend, synchronize
import FastIsostasy: dxx_kernel!, dyy_kernel!, dx_kernel!, dy_kernel!, dxx_dyy_dx_kernel!

# ─── Setup ───────────────────────────────────────────────────────────────────
W = 3000f3
n = 8   # 256×256; change to 9 for 512×512, 10 for 1024×1024

domain = RegionalDomain(W, n)
T = eltype(domain.x)
nx, ny = domain.nx, domain.ny

u   = rand(T, nx, ny)
uxx = similar(u)
uyy = similar(u)
ux  = similar(u)
uxy = similar(u)

backend = get_backend(u)

println("Grid: $(nx)×$(ny)  (n=$n, T=$T)\n")

# ─── CPU pure-loop baseline (derivatives.jl) ─────────────────────────────────
# update_second_derivatives! dispatches to the Matrix{T} method:
# dxx!(uxx, u, domain) → dyy!(uyy, u, domain) → dxy!(ux, uxy, u, domain)
# Total: 4 passes over the arrays.
println("=== CPU pure loops ===")
tmp = @be update_second_derivatives!($uxx, $uyy, $ux, $uxy, $u, $domain)
display(tmp)

# ─── KA unfused (4 separate kernel launches) ─────────────────────────────────
println("\n=== KA unfused (4 kernel launches) ===")
tmp = @be begin
    dxx_kernel!($backend)($uxx, $u, $domain.Dx; ndrange=($nx, $ny))
    synchronize($backend)
    dyy_kernel!($backend)($uyy, $u, $domain.Dy; ndrange=($nx, $ny))
    synchronize($backend)
    dx_kernel!($backend)($ux, $u, $domain.Dx; ndrange=($nx, $ny))
    synchronize($backend)
    dy_kernel!($backend)($uxy, $ux, $domain.Dy; ndrange=($nx, $ny))
    synchronize($backend)
end
display(tmp)

# ─── KA fused (2 kernel launches) ────────────────────────────────────────────
# dxx_dyy_dx_kernel! computes dxx, dyy, and dx in a single pass;
# dy_kernel! then completes dxy in a second pass.
println("\n=== KA fused (2 kernel launches) ===")
tmp = @be begin
    dxx_dyy_dx_kernel!($backend)($uxx, $uyy, $ux, $u, $domain.Dx, $domain.Dy; ndrange=($nx, $ny))
    synchronize($backend)
    dy_kernel!($backend)($uxy, $ux, $domain.Dy; ndrange=($nx, $ny))
    synchronize($backend)
end
display(tmp)

# ─── GPU benchmarks ───────────────────────────────────────────────────────────
if CUDA.functional()
    gpu_domain = RegionalDomain(W, n; arraykernel = CuArray)
    u_gpu   = CuArray(u)
    uxx_gpu = similar(u_gpu)
    uyy_gpu = similar(u_gpu)
    ux_gpu  = similar(u_gpu)
    uxy_gpu = similar(u_gpu)
    gpu_backend = get_backend(u_gpu)

    println("\n=== GPU KA unfused (4 kernel launches) ===")
    tmp = @be CUDA.@sync begin
        dxx_kernel!($gpu_backend)($uxx_gpu, $u_gpu, $gpu_domain.Dx; ndrange=($nx, $ny))
        dyy_kernel!($gpu_backend)($uyy_gpu, $u_gpu, $gpu_domain.Dy; ndrange=($nx, $ny))
        dx_kernel!($gpu_backend)($ux_gpu, $u_gpu, $gpu_domain.Dx; ndrange=($nx, $ny))
        dy_kernel!($gpu_backend)($uxy_gpu, $ux_gpu, $gpu_domain.Dy; ndrange=($nx, $ny))
    end
    display(tmp)

    println("\n=== GPU KA fused / update_second_derivatives! (2 kernel launches) ===")
    tmp = @be CUDA.@sync begin
            dxx_dyy_dx_kernel!($gpu_backend)($uxx_gpu, $uyy_gpu, $ux_gpu, $u_gpu, $gpu_domain.Dx, $gpu_domain.Dy; ndrange=($nx, $ny))
            dy_kernel!($gpu_backend)($uxy_gpu, $ux_gpu, $gpu_domain.Dy; ndrange=($nx, $ny))
    end
    display(tmp)
else
    println("\nNo functional CUDA device found, skipping GPU benchmarks.")
end