
push!(LOAD_PATH, "../")
using FastIsostasy
using Test

T = Float64
n = 6
W = T(2000e3)               # half-length of the square domain (m)
Omega = ComputationDomain(W, n)   # domain parameters
R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)

eta_channel = fill(1e21, size(Omega.X)...)
p = LateralVariability(Omega)
c = PhysicalConstants(T)

timespan = T.([0, 1e4]) * T(c.seconds_per_year)     # (yr) -> (s)
dt = T(100) * T(c.seconds_per_year)                 # (yr) -> (s)
t_vec = timespan[1]:dt:timespan[2]                  # (s)

u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
u3D_elastic = copy(u3D)
u3D_viscous = copy(u3D)
dudt3D_viscous = copy(u3D)

domains = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)

sigma_zz_zero = copy(u3D[:, :, 1])          # test a zero-load field (N/m^2)
tools = precompute_terms(dt, Omega, p, c)

@testset "symmetry of load response" begin
    @test isapprox( tools.elasticgreen, tools.elasticgreen', rtol = T(1e-6) )
    @test isapprox( tools.elasticgreen, reverse(tools.elasticgreen, dims=1), rtol = T(1e-6) )
    @test isapprox( tools.elasticgreen, reverse(tools.elasticgreen, dims=2), rtol = T(1e-6) )
end

@testset "homogeneous response to zero load" begin
    @time forward_isostasy!(
        Omega,
        t_out,
        u3D_elastic,
        u3D_viscous,
        dudt3D_viscous,
        sigma_zz,
        tools,
        p,
        c,
    )
    @test sum( isapprox.(u3D_elastic, T(0)) ) == prod(size(u3D))
    @test sum( isapprox.(u3D_viscous, T(0)) ) == prod(size(u3D))
end