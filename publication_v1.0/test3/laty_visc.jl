using DelimitedFiles, CairoMakie, Interpolations, FastIsostasy
include("../helpers.jl")

prefix = "../data/Latychev"
files = ["$prefix/stp_NPBm", "$prefix/stp_NPBp"]
lon = 0:0.5:360
lat = -90:0.5:90
data = vec(readdlm(files[2], skipstart = 2))
r = data[1:26]
eta_anom_vec = data[27:end]
nlon, nlat, nr = length(lon), length(lat), length(r)
eta_anom = reshape(eta_anom_vec, (nlon, nlat, nr))
heatmap(eta_anom[:, :, end])
lon180, eta_anom180 = lon360tolon180(lon, eta_anom)
eta_anom_itp = linear_interpolation((lon180, lat, r), eta_anom180, extrapolation_bc = Flat())


Omega = ComputationDomain(3000e3, 7)
eta_anom_ant = eta_anom_itp.(Omega.Lon, Omega.Lat, r[10])

using LinearAlgebra
include("../../test/helpers/compute.jl")

W = (Omega.Wx + Omega.Wy) / 2
sigma = diagm([(W/4)^2, (W/4)^2])
eta_anom_soll = generate_gaussian_field(Omega, 0.0, [0.0, 0.0], 1.0, sigma)

fig = Figure(resolution = (2000, 1000))
axs = [Axis(fig[1, j], aspect = DataAspect()) for j in 1:2]
heatmap!(axs[1], eta_anom_ant, colorrange = (-1, 1))
heatmap!(axs[2], eta_anom_soll, colorrange = (-1, 1))
fig