push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie

function mask_disc(X::Matrix{T}, Y::Matrix{T}, R::T) where {T<:AbstractFloat}
    return T.(X .^ 2 + Y .^ 2 .< R^2)
end

function generate_uniform_disc_load(
    d::DomainParams,
    c::PhysicalConstants,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    D = mask_disc(d.X, d.Y, R)
    return -D .* (c.rho_ice * c.g * H)
end

# function main()
T = Float32

p = init_solidearth_params(T)
c = init_physical_constants(T)

timespan = T.([0, 1e3]) * T(c.seconds_per_year)     # (s)
dt = T(1) * T(c.seconds_per_year)                   # (s)
t_vec = timespan[1]:dt:timespan[2]                  # (s)

L = T(2e6)              # half-length of the square domain (m)
N = 2^8                 # number of cells on domain (1)
d = init_domain(L, N)   # domain parameters

u3D = zeros( T, (size(d.X)..., length(t_vec)) )
R = T(1000e3)
H = T(1000)
sigma_zz_zero = copy(u3D[:, :, 1])   # first, test a zero-load field (N/m^2)
sigma_zz_disc = generate_uniform_disc_load(d, c, R, H)

tools = init_integrator_tools(dt, u3D[:, :, 1], d, p, c)
@time forward_isostasy!(t_vec, u3D, sigma_zz_disc, tools)

ncols = 2
fig = Figure(resolution=(1600, 900))
ax1 = Axis(fig[1, 1])
ax2 = Axis3(fig[1, 2])
hm = heatmap!(ax1, sigma_zz_disc)
sf = surface!(ax2, d.X, d.Y, u3D[:,:,end])
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
save("example_deformation.png", fig)
# end

# main()