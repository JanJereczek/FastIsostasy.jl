push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

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

(_, _, _), _, Titp = load_lithothickness_pan2022()
Tlitho = Titp.(Lon, Lat) .* 1e3

lb = collect(100e3:100e3:500e3)
rlb = c.r_equator .- lb
(_, _, _), _, eta_itp = load_3Dvisc_pan2022()
lv_3D = 10 .^ cat([eta_itp.(Lon, Lat, rlb[k]) for k in eachindex(rlb)]..., dims=3)
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