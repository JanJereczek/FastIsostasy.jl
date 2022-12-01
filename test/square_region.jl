push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using DelimitedFiles
using SpecialFunctions
include("helpers.jl")

# function main()
T = Float64

p = init_solidearth_params(T)
c = init_physical_constants(T)

timespan = T.([0, 1e4]) * T(c.seconds_per_year)     # (yr) -> (s)
dt = T(10) * T(c.seconds_per_year)                 # (yr) -> (s)
t_vec = timespan[1]:dt:timespan[2]                  # (s)

L = T(2000e3)               # half-length of the square domain (m)
n = 7                       # 2^n+1 cells on domain (1)
Omega = init_domain(L, n)   # domain parameters
R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)

u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
u3D_elastic = copy(u3D)
u3D_viscous = copy(u3D)

@testset "analytic solution" begin
    sol = analytic_solution(T(0), T(50000 * c.seconds_per_year), c, p, H, R)
    @test isapprox( sol, -1000*c.rho_ice/p.rho_mantle, rtol=T(1e-2) )
end

plot_analytical_sol = false

if plot_analytical_sol
    analytic_solution_r(r) = analytic_solution(r, T(50000 * c.seconds_per_year), c, p, H, R, n_quad_support = 100_000)
    u_analytic = analytic_solution_r.( sqrt.(Omega.X .^ 2 + Omega.Y .^ 2) )

    fig = Figure(resolution = (1600, 900))
    ax = Axis(fig[1, 1], aspect = AxisAspect(1))
    ax3 = Axis3(fig[1, 3])
    hidedecorations!(ax)

    hm = heatmap!(ax, u_analytic, colorrange = (-300, 300), colormap = :balance)
    wireframe!(ax3, Omega.X, Omega.Y, u_analytic)
    Colorbar(fig[1, 2], hm, label = L"Bedrock displacement $u$ (m)", height = Relative(.5))
end

sigma_zz_zero = copy(u3D[:, :, 1])   # first, test a zero-load field (N/m^2)
sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
tools = init_integrator_tools(dt, Omega, p, c)

@testset "symmetry of load response" begin
    @test isapprox( tools.loadresponse, tools.loadresponse', rtol = T(1e-6) )
    @test isapprox( tools.loadresponse, reverse(tools.loadresponse, dims=1), rtol = T(1e-6) )
    @test isapprox( tools.loadresponse, reverse(tools.loadresponse, dims=2), rtol = T(1e-6) )
end

# @testset "similarity to matlab code" begin
#     matlab_integrated_green = readdlm("data/integrated_green.txt", ',', T)
#     matlab_fft_integrated_green = readdlm("data/fft_integrated_green.txt", ',', Complex{T})

#     approx_green = isapprox.(tools.loadresponse, matlab_integrated_green, rtol=T(5e-2))
#     @test sum(approx_green) == prod(size(tools.loadresponse) )

#     # @test sum( isapprox.( tools.fourier_loadresponse, matlab_fft_integrated_green, rtol = T(5e-2) ) ) == prod(size(tools.loadresponse))
# end

@testset "homogenous response to zero load" begin
    forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz_zero, tools, c)
    @test sum( isapprox.(u3D_elastic, T(0)) ) == prod(size(u3D))
    @test sum( isapprox.(u3D_viscous, T(0)) ) == prod(size(u3D))
end

@time forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz_disc, tools, c)

ncols = 3
fig = Figure(resolution=(1600, 900))
ax1 = Axis(fig[1, 1], aspect=AxisAspect(1))
hm = heatmap!(ax1, sigma_zz_disc)
Colorbar(
    fig[2,1],
    hm,
    label = L"Vertical load $ \mathrm{N \, m^{-2}}$",
    vertical = false,
    width = Relative(0.8),
)

u_plot = [ u3D_elastic[:,:,end], u3D_viscous[:,:,end], u3D_elastic[:,:,end] + u3D_viscous[:,:,end] ]
labels = [
    L"Vertical displacement of elastic response $u^E$ (m)",
    L"Vertical displacement of viscous response $u^V$ (m)",
    L"Total vertical displacement $u^E + u^V$ (m)",
]
for j in eachindex(u_plot)
    ax3D = Axis3(fig[1, j+1])
    sf = surface!(ax3D, Omega.X, Omega.Y, u_plot[j])
    Colorbar(
        fig[2, j+1],
        sf,
        label = labels[j],
        vertical = false,
        width = Relative(0.8),
    )
end
save("plots/example_deformation.png", fig)
# end

# main()