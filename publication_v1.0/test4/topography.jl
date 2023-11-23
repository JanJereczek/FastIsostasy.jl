using NCDatasets, DelimitedFiles, CairoMakie, LinearAlgebra
include("../../test/helpers/compute.jl")

function load_topo(file, nlon, nlat)
    topovec = readdlm(file)
    topo = reshape(topovec, nlon, nlat)
    reverse!(topo, dims = 2)
    idx = vcat(nlon÷2+1:nlon, 1:nlon÷2)
    return topo[idx, :]
end

function load_latychev_topo(; case = "3D")
    if case == "1D"
        file = "../data/Latychev/Topography/Topo_REF/T_000"
    elseif case == "3D"
        file = "../data/Latychev/Topography/Topo_SMD/T_000"
    else
        error("Provided case does not exist.")
    end
    lonlat256, _ = readdlm("../data/Latychev/ICE6G/gl256_LatLon", header = true)
    lon256 = reshape(lonlat256[:, 1], 512, 256)
    dlon256 = mean(diff(lon256[:, 1]))
    dlon512 = dlon256/2
    nlon, nlat = 1024, 512
    lon = -180:dlon512:180-dlon512
    lat = -90+dlon512/2:dlon512:90-dlon512/2
    topo = load_topo(file, nlon, nlat)
    itp = linear_interpolation((lon, lat), topo, extrapolation_bc = Flat())
    return (lon, lat), topo, itp
end

# nlon, nlat = 1024, 512
# lon = -180:dlon512:180-dlon512
# lat = -90+dlon512/2:dlon512:90-dlon512/2
# topo1D = load_topo(file1D, nlon, nlat)
# topo3D = load_topo(file1D, nlon, nlat)

#=
(lon, lat), topo, itp = load_laty_topo()
heatmap(topo)
Omega = ComputationDomain(3500e3, 3500e3, 128, 128)
topo_ant = itp.(Omega.Lon, Omega.Lat)
lim = 3e6
mask = (topo_ant .> -2000) .& (-lim .< Omega.X .< lim) .& (-lim .< Omega.Y .< lim)
heatmap(mask)
sigma = diagm([(Omega.Wx/20)^2, (Omega.Wy/20)^2])
kernel = generate_gaussian_field(Omega, 0.0, [0.0, 0.0], 1.0, sigma)
heatmap(kernel)
filtmask = copy(samesize_conv(kernel, Float64.(mask), Omega))
heatmap(filtmask .> 0.5 * maximum(filtmask))
=#