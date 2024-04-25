@testset "ocean surface" begin
    osc = OceanSurfaceChange()
    z = -150:0.1:70
    srf = osc.A_itp.(z)

    Vice = 45e6 * (1e3)^3
    nV = 10000
    dV = Vice / nV
    Vvec = -dV:-dV:-Vice
    zvec = zeros(nV)
    osc = OceanSurfaceChange()
    for i in 1:nV
        osc(-dV)
        zvec[i] = osc.z_k
    end
    @test osc.z_k < 0.0
    @test osc.A_k < osc.A_itp(0.0)
end