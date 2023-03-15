push!(LOAD_PATH, "../")
using FastIsostasy
using Test

lat, lon = -85.4, 67.8
lat0, lon0 = -71.0, 0.0
k, x, y = latlon2stereo(lat, lon, lat0, lon0)
lat_, lon_ = stereo2latlon(x, y, lat0, lon0)
@test lat_ ≈ lat
@test lon_ ≈ lon

latvec = -90.0:1.0:-60.0
lonvec = -179.0:1.0:180.0
Lon, Lat = meshgrid(lonvec, latvec)
K, X, Y = latlon2stereo(Lat, Lon, lat0, lon0)
Lat_, Lon_ = stereo2latlon(X, Y, lat0, lon0)
@test Lat ≈ Lat_
@test Lon[2, :] ≈ Lon_[2, :]    # singularity at lat = -90°