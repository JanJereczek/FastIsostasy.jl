push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using DelimitedFiles
using SpecialFunctions
include("helpers.jl")

# function main()
T = Float32

p = init_solidearth_params(T)
c = init_physical_constants(T)

timespan = T.([0, 1e3]) * T(c.seconds_per_year)     # (s)
dt = T(1) * T(c.seconds_per_year)                   # (s)
t_vec = timespan[1]:dt:timespan[2]                  # (s)

L = T(2000e3)               # half-length of the square domain (m)
N = 2^6                     # number of cells on domain (1)
Omega = init_domain(L, N)   # domain parameters

u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)

@testset "analytic solution" begin
    sol = analytic_solution(T(0), T(50000 * c.seconds_per_year), c, p, H, R)
    @test isapprox( sol, -1000*c.rho_ice/p.rho_mantle, rtol=T(1e-2) )
end

sigma_zz_zero = copy(u3D[:, :, 1])   # first, test a zero-load field (N/m^2)
sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
tools = init_integrator_tools(dt, Omega, p, c)

@testset "symmetry of load response" begin
    @test isapprox( tools.loadresponse, tools.loadresponse', rtol = T(1e-6) )
    @test isapprox( tools.loadresponse, reverse(tools.loadresponse, dims=1), rtol = T(1e-6) )
    @test isapprox( tools.loadresponse, reverse(tools.loadresponse, dims=2), rtol = T(1e-6) )
end

@testset "similarity to matlab code" begin
    matlab_integrated_green = readdlm("data/integrated_green.txt", ',', T)
    matlab_fft_integrated_green = readdlm("data/fft_integrated_green.txt", ',', Complex{T})

    approx_green = isapprox.(tools.loadresponse, matlab_integrated_green, rtol=T(5e-2))
    @test sum(approx_green) == prod(size(tools.loadresponse) )

    # @test sum( isapprox.( tools.fourier_loadresponse, matlab_fft_integrated_green, rtol = T(5e-2) ) ) == prod(size(tools.loadresponse))
end

@testset "homogenous response to zero load" begin
    forward_isostasy!(t_vec, u3D, sigma_zz_zero, tools, c)
    @test sum( isapprox.(u3D, T(0)) ) == prod(size(u3D))
end


@time forward_isostasy!(t_vec, u3D, sigma_zz_disc, tools, c)

ncols = 2
fig = Figure(resolution=(1600, 900))
ax1 = Axis(fig[1, 1])
ax2 = Axis3(fig[1, 2])
hm = heatmap!(ax1, sigma_zz_disc)
sf = surface!(ax2, Omega.X, Omega.Y, u3D[:,:,end])
Colorbar(
    fig[2,1],
    hm,
    label = L"Vertical load $ \mathrm{N \, m^{-2}}$",
    vertical = false,
    width = Relative(0.8),
)
Colorbar(
    fig[2,2],
    sf,
    label = L"Vertical displacement $ \mathrm{m}$",
    vertical = false,
    width = Relative(0.8),
)
save("plots/example_deformation.png", fig)
# end

# main()