@testset "ELRA" begin
    n = 6
    use_cuda = false
    dense = false

    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n, use_cuda = use_cuda, correct_distortion = false)
    c = PhysicalConstants(rho_litho = 0.0)
    p = LayeredEarth(Omega, tau = years2seconds(3e3), layer_boundaries = [100e3, 600e3])

    opts = SolverOptions(verbose = true, deformation_model = :elra)
    fip = FastIsoProblem(Omega, c, p, zeros(2), zeros(2),
        [Omega.null, Omega.null], opts = opts, output = "sparse")

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hice = uniform_ice_cylinder(Omega, R, H)
    update_loadcolumns!(fip, Hice)
    columnanom_load!(fip)
    update_deformation_rhs!(fip, fip.now.u)
    samesize_conv!(fip.now.u_eq, fip.tools.prealloc.convo_out,
        - (fip.now.columnanoms.load + fip.now.columnanoms.litho) .* fip.c.g .*
        fip.Omega.K .^ 2, fip.tools.viscousconvo, fip.Omega)
    @test maximum(fip.now.u_eq) < 10
    @test minimum(fip.now.u_eq) > -300
end