push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using Test
using SpecialFunctions
using JLD2
using Interpolations
include("helpers.jl")

# function main()
T = Float64

p = init_solidearth_params(T)
c = init_physical_constants(T)

timespan = T.([0, 1e4]) * T(c.seconds_per_year)     # (yr) -> (s)
dt = T(100) * T(c.seconds_per_year)                 # (yr) -> (s)
t_vec = timespan[1]:dt:timespan[2]                  # (s)

L = T(2000e3)               # half-length of the square domain (m)
n = 8                       # 2^n+1 cells on domain (1)
Omega = init_domain(L, n)   # domain parameters
R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)

u3D = zeros( T, (size(Omega.X)..., length(t_vec)) )
u3D_elastic = copy(u3D)
u3D_viscous = copy(u3D)
domains = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)

@testset "analytic solution" begin
    sol = analytic_solution(T(0), T(50000 * c.seconds_per_year), c, p, H, R, domains)
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

@testset "homogenous response to zero load" begin
    forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz_zero, tools, c)
    @test sum( isapprox.(u3D_elastic, T(0)) ) == prod(size(u3D))
    @test sum( isapprox.(u3D_viscous, T(0)) ) == prod(size(u3D))
end

@time forward_isostasy!(Omega, t_vec, u3D_elastic, u3D_viscous, sigma_zz_disc, tools, c)
jldsave(
    "data/numerical_solution_N$(Omega.N).jld2",
    u3D_elastic = u3D_elastic,
    u3D_viscous = u3D_viscous,
)
##############

# Computing analytical solution is quite expensive as it involves
# integration over κ ∈ [0, ∞) --> load precomputed interpolator.
compute_analytical_sol = false
tend = T(Inf * c.seconds_per_year)

if compute_analytical_sol
    analytic_solution_r(r) = analytic_solution(r, tend, c, p, H, R, domains)
    u_analytic = analytic_solution_r.( sqrt.(Omega.X .^ 2 + Omega.Y .^ 2) )
    U = [u_analytic[i,j] for i in axes(Omega.X, 1), j in axes(Omega.X, 2)]
    u_analytic_interp = linear_interpolation(
        (Omega.X[1,:], Omega.Y[:,1]),
        u_analytic,
        extrapolation_bc = NaN,
    )
    jldsave(
        "data/analytical_solution_interpolator_N$(Omega.N).jld2",
        u_analytic_interp = u_analytic_interp,
    )
end
u_analytic_interp = load(
    "data/analytical_solution_interpolator_N65.jld2",
    "u_analytic_interp",
)
u_analytic = u_analytic_interp.(Omega.X, Omega.Y)

nrows = 2
ncols = 3
fig = Figure(resolution=(1600, 900))

ax1 = Axis(fig[1, 1][1, :], aspect=DataAspect())
hm = heatmap!(ax1, Omega.X, Omega.Y, sigma_zz_disc)
Colorbar(
    fig[1, 1][2, :],
    hm,
    label = L"Vertical load $ \mathrm{N \, m^{-2}}$",
    vertical = false,
    width = Relative(0.8),
)

u_plot = [
    u3D_elastic[:,:,end],
    u3D_elastic[:,:,end] + u3D_viscous[:,:,end],
    u3D_viscous[:,:,end],
    u_analytic,
]
panels = [
    (1,2),
    (1,3),
    (2,2),
    (2,3),
]
labels = [
    L"Vertical displacement of elastic response $u^E$ (m)",
    L"Total vertical displacement $u^E + u^V$ (m)",
    L"Vertical displacement of viscous response $u^V$ (m)",
    L"Analytical solution for total displacement $u^A$ (m)",
]
for k in eachindex(u_plot)
    i, j = panels[k]
    ax3D = Axis3(fig[i, j][1, :])
    sf = surface!(
        ax3D,
        Omega.X,
        Omega.Y,
        u_plot[k],
        colorrange = (-300, 50),
        colormap = :jet,
    )
    wireframe!(
        ax3D,
        Omega.X,
        Omega.Y,
        u_plot[k],
        linewidth = 0.1,
        color = :black,
    )
    Colorbar(
        fig[i, j][2, :],
        sf,
        label = labels[k],
        vertical = false,
        width = Relative(0.8),
    )
end
plotname = "plots/discload_deformation_N$(Omega.N)"
save("$plotname.png", fig)
save("$plotname.pdf", fig)


# end

# main()