using FastIsostasy
using JLD2, NCDatasets, Interpolations, DelimitedFiles

# (lon, lat, tlaty), sl, itp = load_latychev2023_ICE6G(case = "3D", var = "SL")
# Lon, Lat = meshgrid(lon, lat[1:40])
# gmslvec = [mean(sl[:, :, k]) for k in axes(sl, 3)]
# shmslvec = [mean(itp.(Lon, Lat, t)) for t in tlaty]

function convert_laty2023()
    T = Float32
    _, sl, _ = load_latychev2023_ICE6G(case = "3D", var = "SL")
    _, g, _ = load_latychev2023_ICE6G(case = "3D", var = "G")
    (lon, lat, t), r, _ = load_latychev2023_ICE6G(case = "3D", var = "R")
    # ds = NCDataset("/home/jan/pCloudSync/PhD/Projects/Isostasy/IsostasyData/"*
    #     "Seakon3D-ICE6G.nc","c")
    ds = NCDataset("Seakon3D-ICE6G.nc","c")
    defDim(ds, "lon", length(lon))
    defDim(ds, "lat", length(lat))
    defDim(ds, "time", length(t))
    x1 = defVar(ds, "lon", T, ("lon",))
    x1[:] = T.(lon)
    x2 = defVar(ds, "lat", T, ("lat",))
    x2[:] = T.(lat)
    x3 = defVar(ds, "time", T, ("time",))
    x3[:] = T.(t)

    nc1 = defVar(ds, "sea level", T, ("lon", "lat", "time"))
    nc1[:, :, :] = T.(sl)
    nc1.attrib["units"] = "meters"

    nc2 = defVar(ds, "geoid", T, ("lon", "lat", "time"))
    nc2[:, :, :] = T.(g)
    nc2.attrib["units"] = "meters"

    nc3 = defVar(ds, "vertical displacement", T, ("lon", "lat", "time"))
    nc3[:, :, :] = T.(r)
    nc3.attrib["units"] = "meters"

    close(ds)
    return nothing
end

convert_laty2023()