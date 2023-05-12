push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, CairoMakie
include("helpers_compute.jl")
include("external_viscosity_maps.jl")

function get_wiens_layervisc(Omega)
    halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)
    Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    return lv
end

n = 5
T = Float64
L = T(3000e3)               # half-length of the square domain (m)
Omega = ComputationDomain(L, n)
c = PhysicalConstants()

lb = [88e3, 180e3, 280e3, 400e3]
lv = get_wiens_layervisc(Omega)
p = MultilayerEarth(
    Omega,
    c,
    layers_begin = lb,
    layers_viscosity = lv,
)
ground_truth = copy(p.effective_viscosity)

R = T(2000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.(0.0:1_000.0:2_000.0)

t1 = time()
results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3(), active_geostate=false)
t_fastiso = time() - t1
println("Took $t_fastiso seconds!")
println("-------------------------------------")

tinv = t_out[2:end]
Hice = [Hcylinder for t in tinv]
Y = results.viscous[2:end]
paraminv = ParamInversion(Omega, c, p, tinv, Y, Hice)
priors, ukiobj = perform(paraminv)
logeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, paraminv)

titles = [L"True viscosity field $\,$", L"Estimated viscosity field $\,$"]
cmap = cgrad(:jet, rev=true)
ncols = length(titles)
fig = Figure(resolution = (1600, 900), fontsize = 28)
axs = [Axis(fig[1, j], aspect = DataAspect(), title=titles[j]) for j in 1:ncols]
[hidedecorations!(axs[j]) for j in 1:ncols]

x = 1:Omega.N
heatmap!(axs[1], x, x, log10.(ground_truth), colorrange = (20, 21), colormap = cmap)
contour!(axs[1], x .+ 0.5, x .+ 0.5, paraminv.obs_idx, levels = [0.99], color = :white, linewidth = 5)

heatmap!(axs[2], x, x, log10.(paraminv.p.effective_viscosity), colorrange = (20, 21), colormap = cmap)
contour!(axs[2], x .+ 0.5, x .+ 0.5, paraminv.obs_idx, levels = [0.99], color = :white, linewidth = 5)

Colorbar(fig[2, :], colorrange = (20, 21), colormap = cmap, vertical = false, width = Relative(0.5))
save("plots/test6/UKI.pdf", fig)
save("plots/test6/UKI.png", fig)