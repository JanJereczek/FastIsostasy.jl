push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

(lon, lat, tlaty), sl, itp = load_latychev2023_ICE6G(case = "3D", var = "SL")
Lon, Lat = meshgrid(lon, lat[1:40])
gmslvec = [mean(sl[:, :, k]) for k in axes(sl, 3)]
shmslvec = [mean(itp.(Lon, Lat, t)) for t in tlaty]
