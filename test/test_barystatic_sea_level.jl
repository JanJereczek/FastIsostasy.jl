@testset "barystatic sea level" begin
    bsl = PiecewiseConstantBSL()
    Vice = 45e6 * (1e3)^3
    nV = 10000
    dV = Vice / nV
    Vvec = -dV:-dV:-Vice
    zvec = zeros(nV)
    for i in 1:nV
        update_bsl!(bsl, -dV, 0.0)
    end
    @test bsl.z < -120.0
    @test bsl.A < bsl.ref.A
end