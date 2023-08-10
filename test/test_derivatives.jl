Omega = ComputationDomain(3000e3, 6)
function benchmark1_constants(Omega)
    c = PhysicalConstants(rho_litho = 0.0)
    p = LateralVariability(Omega)
    R, H = 1000e3, 1e3
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    interactive_sealevel = false
    return c, p, R, H, Hcylinder, t_out, interactive_sealevel
end
c, p, R, H, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)

alpha = 3
beta = 2
Xlin = alpha .* Omega.X
Xquad = Omega.X .^ beta

dx!(fip.tools.dummies.uxx, Xlin, Omega)
fip.tools.dummies.uxx[2:end-1, 2:end-1] ≈ alpha ./ Omega.K[2:end-1, 2:end-1]
dx!(fip.tools.dummies.uxx, Xquad, Omega)
fip.tools.dummies.uxx[2:end-1, 2:end-1] ≈ beta .* Omega.X[2:end-1, 2:end-1] ./ Omega.K[2:end-1, 2:end-1]
dxx!(fip.tools.dummies.uxx, Xlin, Omega)
fip.tools.dummies.uxx ≈ Omega.null
dxx!(fip.tools.dummies.uxx, Xquad, Omega)
fip.tools.dummies.uxx[2:end-1, 2:end-1] ≈ beta ./ Omega.K[2:end-1, 2:end-1].^2

dXquad_test = (Xquad[3:end, :] - 2 .* Xquad[2:end-1, :] + Xquad[1:end-2, :]) ./ (Omega.K[2:end-1, :] .* Omega.dx).^2

du = copy(Omega.null)
u = Xquad
@inbounds for j in axes(du, 2)
    for i in axes(du, 1)[2:Omega.Nx-1]
        du[i, j] = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (Omega.K[i, j] * Omega.dx)^2
    end
    du[1, j] = du[2, j]
    du[Omega.Nx, j] = du[Omega.Nx-1, j]
end