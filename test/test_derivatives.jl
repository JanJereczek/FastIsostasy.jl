# Omega = ComputationDomain(3000e3, 6)
# function benchmark1_constants(Omega)
#     c = PhysicalConstants(rho_litho = 0.0)
#     p = LateralVariability(Omega)
#     R, H = 1000e3, 1e3
#     Hcylinder = uniform_ice_cylinder(Omega, R, H)
#     t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
#     interactive_sealevel = false
#     return c, p, R, H, Hcylinder, t_out, interactive_sealevel
# end
# c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
# fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)

# alpha = 3
# beta = 2
# Xlin = alpha .* Omega.X
# Xquad = Omega.X .^ beta

# dx!(fip.tools.prealloc.uxx, Xlin, Omega)
# fip.tools.prealloc.uxx[2:end-1, 2:end-1] ≈ alpha ./ Omega.K[2:end-1, 2:end-1]
# dx!(fip.tools.prealloc.uxx, Xquad, Omega)
# fip.tools.prealloc.uxx[2:end-1, 2:end-1] ≈ beta .* Omega.X[2:end-1, 2:end-1] ./ Omega.K[2:end-1, 2:end-1]
# dxx!(fip.tools.prealloc.uxx, Xlin, Omega)
# fip.tools.prealloc.uxx ≈ Omega.null
# dxx!(fip.tools.prealloc.uxx, Xquad, Omega)
# fip.tools.prealloc.uxx[2:end-1, 2:end-1] ≈ beta ./ Omega.K[2:end-1, 2:end-1].^2

# dXquad_test = (Xquad[3:end, :] - 2 .* Xquad[2:end-1, :] + Xquad[1:end-2, :]) ./ (Omega.K[2:end-1, :] .* Omega.dx).^2

# du = copy(Omega.null)
# u = Xquad
# @inbounds for j in axes(du, 2)
#     for i in axes(du, 1)[2:Omega.Nx-1]
#         du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (Omega.K[i, j] * Omega.dx)^2
#     end
#     du[1, j] = du[2, j]
#     du[Omega.Nx, j] = du[Omega.Nx-1, j]
# end

function xpu_derivative_equivalence()

    W, n, pc = 3000e3, 7, false
    Omega = ComputationDomain(W, n, projection_correction = pc)
    c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)
    XY = copy(Omega.X .^ 2 .* Omega.Y .^ 2)

    Omega_gpu = ComputationDomain(W, n, projection_correction = pc, use_cuda = true)
    c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega_gpu)
    fip_gpu = FastIsoProblem(Omega_gpu, c, p, t_out, interactive_sealevel, Hcylinder)
    XY_gpu = copy(Omega_gpu.X .^ 2 .* Omega_gpu.Y .^ 2)

    P = fip.tools.prealloc
    P_gpu = fip_gpu.tools.prealloc

    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, XY, Omega)
    update_second_derivatives!(P_gpu.uxx, P_gpu.uyy, P_gpu.ux, P_gpu.uxy, XY_gpu, Omega_gpu)
    
    uxx = 2 .* Omega.Y .^ 2
    uyy = 2 .* Omega.X .^ 2
    uxy = 4 .* Omega.X .* Omega.Y
    @test inn(P.uxx) ≈ inn(uxx)
    @test inn(Array(P_gpu.uxx)) ≈ inn(uxx)
    @test inn(P.uyy) ≈ inn(uyy)
    @test inn(Array(P_gpu.uyy)) ≈ inn(uyy)
    @test inn(P.uxy) ≈ inn(uxy)
    @test inn(Array(P_gpu.uxy)) ≈ inn(uxy)
end

function inn(X)
    return X[2:end-1, 2:end-1]
end