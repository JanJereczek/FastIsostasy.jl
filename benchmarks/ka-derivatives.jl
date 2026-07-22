using FastIsostasy
using LoopVectorization: @turbo
using Chairmarks
using Printf
using KernelAbstractions: get_backend, synchronize

import FastIsostasy: dxx!, dyy!, dx!, dy!
import FastIsostasy: dxx_kernel!, dyy_kernel!, dx_kernel!, dy_kernel!, dxx_dyy_dx_kernel!

# ─── Setup ────────────────────────────────────────────────────────────────────
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

fmt(b) = let t = minimum(b).time * 1e6
    "$(round(t, digits=1)) µs"
end
speedup(base, fast) = round(minimum(base).time / minimum(fast).time, digits=2)

println("Grid: $(nx)×$(ny)  (n=$n, T=$T)\n")

# ─── Baseline: @inbounds, original loop order (no @turbo) ─────────────────────
# dyy! and dy! had outer-i/inner-j order (non-contiguous inner access in
# column-major layout). Kept here verbatim as the reference.

function dxx_base!(du, u, domain)
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:domain.nx-1]
            du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (domain.Dx[i, j] ^ 2)
        end
        du[1, j] = (u[3, j] - 2*u[2, j] + u[1, j]) / (domain.Dx[1, j] ^ 2)
        du[domain.nx, j] = (u[domain.nx, j] - 2*u[domain.nx-1, j] + u[domain.nx-2, j]) /
            (domain.Dx[domain.nx, j] ^ 2)
    end
end

function dyy_base!(du, u, domain)
    @inbounds for i in axes(du, 1)       # outer-i / inner-j: non-contiguous inner access
        for j in axes(du, 2)[2:domain.ny-1]
            du[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (domain.Dy[i, j] ^ 2)
        end
        du[i, 1] = (u[i, 3] - 2*u[i, 2] + u[i, 1]) / (domain.Dy[i, 1] ^ 2)
        du[i, domain.ny] = (u[i, domain.ny] - 2*u[i, domain.ny-1] + u[i, domain.ny-2]) /
            (domain.Dy[i, domain.ny] ^ 2)
    end
end

function dx_base!(du, u, domain)
    @inbounds for j in axes(du, 2)
        for i in axes(du, 1)[2:domain.nx-1]
            du[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * domain.Dx[i, j])
        end
        du[1, j] = (u[2, j] - u[1, j]) / domain.Dx[1, j]
        du[domain.nx, j] = (u[domain.nx, j] - u[domain.nx-1, j]) / domain.Dx[domain.nx, j]
    end
end

function dy_base!(du, u, domain)
    @inbounds for i in axes(du, 1)       # outer-i / inner-j: non-contiguous inner access
        for j in axes(du, 2)[2:domain.ny-1]
            du[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * domain.Dy[i, j])
        end
        du[i, 1] = (u[i, 2] - u[i, 1]) / domain.Dy[i, 1]
        du[i, domain.ny] = (u[i, domain.ny] - u[i, domain.ny-1]) / domain.Dy[i, domain.ny]
    end
end

function update_second_derivatives_base!(uxx, uyy, ux, uxy, u, domain)
    dxx_base!(uxx, u, domain)
    dyy_base!(uyy, u, domain)
    dx_base!(ux, u, domain)
    dy_base!(uxy, ux, domain)
end

# ─── Per-function @inbounds vs @turbo ─────────────────────────────────────────
println("─"^60)
println("  @inbounds vs @turbo (min time over samples)")
println("─"^60)
println("  function   baseline      @turbo       speedup")
println("─"^60)

for (name, base_fn!, turbo_fn!, args) in [
    ("dxx!",  dxx_base!,  dxx!,  (uxx, u,  domain)),
    ("dyy!",  dyy_base!,  dyy!,  (uyy, u,  domain)),
    ("dx!",   dx_base!,   dx!,   (ux,  u,  domain)),
    ("dy!",   dy_base!,   dy!,   (uxy, ux, domain)),
]
    tb = @be base_fn!($args...)
    tt = @be turbo_fn!($args...)
    @printf("  %-10s %9s    %9s    %5.2f×\n",
        name, fmt(tb), fmt(tt), speedup(tb, tt))
end

println("─"^60)
tb = @be update_second_derivatives_base!($uxx, $uyy, $ux, $uxy, $u, $domain)
tt = @be update_second_derivatives!($uxx, $uyy, $ux, $uxy, $u, $domain)
@printf("  %-10s %9s    %9s    %5.2f×\n", "combined", fmt(tb), fmt(tt), speedup(tb, tt))
println("─"^60)

# ─── KA unfused (4 separate kernel launches) ──────────────────────────────────
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

# ─── KA fused (2 kernel launches) ─────────────────────────────────────────────
println("\n=== KA fused (2 kernel launches) ===")
tmp = @be begin
    dxx_dyy_dx_kernel!($backend)($uxx, $uyy, $ux, $u, $domain.Dx, $domain.Dy; ndrange=($nx, $ny))
    synchronize($backend)
    dy_kernel!($backend)($uxy, $ux, $domain.Dy; ndrange=($nx, $ny))
    synchronize($backend)
end
display(tmp)

# ─── GPU benchmarks ───────────────────────────────────────────────────────────
# Requires CUDA in the active Julia environment. Run without --project or after
# Pkg.add("CUDA") to enable GPU sections. Skipped here to avoid scope issues
# with CUDA.@sync macro expansion when CUDA is loaded from the global env.
println("\nGPU benchmarks skipped (run without --project to enable).")
