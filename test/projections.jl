push!(LOAD_PATH, "../")
using FastIsostasy
using Test

lat, lon = -85.4, 67.8
k, x, y = latlon2stereo(lat, lon)
lat_, lon_ = stereo2latlon(x, y)
@test lat_ ≈ lat
@test lon_ ≈ lon

latvec = -90.0:1.0:-60.0
lonvec = -179.0:1.0:180.0
Lon, Lat = meshgrid(lonvec, latvec)
K, X, Y = latlon2stereo(Lat, Lon)
Lat_, Lon_ = stereo2latlon(X, Y)
@test Lat ≈ Lat_
@test Lon ≈ Lon_