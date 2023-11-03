push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

function lon360tolon180(lon, X)
    permidx = lon .> 180
    lon180 = vcat(lon[permidx] .- 360, lon[not.(permidx)])
    Hice180 = cat(X[permidx, :, :], X[not.(permidx), :, :], dims=1)
    return lon180, Hice180
end

function load_laty_3Dvisc()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Latychev/ICE6G/dense/stp_model_MH"
    data = vec(readdlm(file, skipstart = 2))
    r = data[1:64]
    eta_ratio_vec = data[65:end]

    lon = collect(range(0, stop = 360, length = 721))
    lat = collect(range(90, stop = -90, length = 361))
    nr, nlat, nlon = length(r), length(lat), length(lon)
    eta_ratio = reshape(eta_ratio_vec, (nlon, nlat, nr))
    heatmap(eta_ratio[:, :, end])
    reverse!(eta_ratio, dims = 2)
    reverse!(lat)
    heatmap(eta_ratio[:, :, end])
    eta1D = [3.16, 3.16, 1.58, 1.58, 0.5, 0.5] .* 1e21
    r_eta1D = [3480000.0, 5173800.9, 5173800.95, 5701000.0, 5701000.05, 6371000.0]
    eta1D_itp = linear_interpolation(r_eta1D, eta1D)
    eta1D_3D = zeros((nlon, nlat, nr))
    for i in axes(eta1D_3D, 1), j in axes(eta1D_3D, 2)
        eta1D_3D[i, j, :] .= eta1D_itp.(r)
    end
    eta = eta1D_3D .* (10 .^ eta_ratio)
    lon180, eta180 = lon360tolon180(lon, eta)
    heatmap(eta180[:, :, end-5])
    eta_itp = linear_interpolation((lon180, lat, r), eta180, extrapolation_bc = Flat())
    return eta_itp
end

function load_litho_thickness_laty()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Latychev/ICE6G/dense/LAB.llz"
    data, head = readdlm(file, header = true)
    Lon_vec, Lat_vec, T_vec = data[:, 1], data[:, 2], data[:, 3]
    lon, lat = unique(Lon_vec), unique(Lat_vec)
    nlon, nlat = length(lon), length(lat)
    Lon = reshape(Lon_vec, nlon, nlat)
    Lat = reshape(Lat_vec, nlon, nlat)
    Tlitho = reshape(T_vec, nlon, nlat)
    reverse!(Tlitho, dims=2)
    reverse!(Lat, dims=2)
    reverse!(lat)
    heatmap(Tlitho)
    lon180, Tlitho180 = lon360tolon180(lon, Tlitho)
    itp = linear_interpolation((lon180, lat), Tlitho180[:, :, 1], extrapolation_bc = Flat())
    return itp
end


N = 700
maxdepth = 300e3
nlayers = 3
use_cuda = false
interactive_sl = false
Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
c = PhysicalConstants()

Lon, Lat, = Omega.Lon, Omega.Lat
lon = Lon[:, 1]
lat = Lat[1, :]
x = Omega.X[:, 1]
y = Omega.Y[1, :]

Titp = load_litho_thickness_laty()
Tlitho = Titp.(Lon, Lat) .* 1e3

lb = collect(100e3:100e3:500e3)
rlb = c.r_equator .- lb
eta_itp = load_laty_3Dvisc()
lv_3D = cat([eta_itp.(Lon, Lat, rlb[k]) for k in eachindex(rlb)]..., dims=3)
eta_lowerbound = 1e16
lv_3D[lv_3D .< eta_lowerbound] .= eta_lowerbound

ds = NCDataset("/home/jan/latyparams.nc","c")    # "c" = create a new file
defDim(ds, "x", length(x))
defDim(ds, "y", length(y))
defDim(ds, "z", length(rlb))
ds.attrib["title"] = "Parameters provided by K. Latychev for testing FastIsostasy on ICE6G_D."

v = defVar(ds, "log10 mantle viscosity", Float64, ("x","y","z"))
v[:,:] = log10.(lv_3D)
v.attrib["units"] = "Pascal seconds"

w = defVar(ds, "lithospheric thickness", Float64, ("x","y"))
w[:,:] = Tlitho
w.attrib["units"] = "meters"

close(ds)