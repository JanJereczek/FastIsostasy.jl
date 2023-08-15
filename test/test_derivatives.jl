function check_derivatives()
    Omega, P, u, uxx, uyy, uxy = derivative_stdsetup(false)
    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, u, Omega)
    @test inn(P.uxx) ≈ inn(uxx)
    @test inn(P.uyy) ≈ inn(uyy)
    @test inn(P.uxy) ≈ inn(uxy)
end

function check_gpu_derivatives()
    Omega, P, u, uxx, uyy, uxy = derivative_stdsetup(true)

end

function derivative_stdsetup(use_cuda::Bool)
    W, n, pc = 3000e3, 7, false
    Omega = ComputationDomain(W, n, projection_correction = pc, use_cuda = use_cuda)
    c, p, _, __, Hcylinder, t_out, interactive_sealevel = benchmark1_constants(Omega)
    fip = FastIsoProblem(Omega, c, p, t_out, interactive_sealevel, Hcylinder)

    u = copy(Omega.X .^ 2 .* Omega.Y .^ 2)
    uxx = 2 .* Omega.Y .^ 2
    uyy = 2 .* Omega.X .^ 2
    uxy = 4 .* Omega.X .* Omega.Y
    return Omega, fip.tools.prealloc, u, uxx, uyy, uxy
end

function check_derivatives(P, u, Omega, uxx, uyy, uxy)
    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, u, Omega)
    @test inn(Array(P.uxx)) ≈ inn(uxx)
    @test inn(Array(P.uyy)) ≈ inn(uyy)
    @test inn(Array(P_gpu.uxy)) ≈ inn(uxy)
end

function inn(X)
    return X[2:end-1, 2:end-1]
end