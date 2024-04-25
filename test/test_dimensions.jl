using Proj

@testset "future use of Proj.jl" begin
    lon, lat = 67.8, -85.4
    lat_ts = -71.0
    forwardproj = Proj.Transformation("EPSG:4326",
        "+proj=stere +lat_0=-90 +lat_ts=$lat_ts", always_xy=true)
    x, y = forwardproj(lon, lat)
    lon_, lat_ = inv(forwardproj)(x, y)
    @test lon_ ≈ lon
    @test lat_ ≈ lat
end

@testset "stereographic projection" begin
    lat, lon = -85.4, 67.8
    lat0, lon0 = -90.0, 0.0   # fails for lat0 != -90°
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
    # @test Lon[2, :] ≈ Lon_[2, :]    # singularity at lat = -90°
end

@testset "ij indexing for xy" begin
    x, y = 1.0:10, 1.0:5
    X, Y = meshgrid(collect(x), collect(y))
    @test X[:, 1] == x
    @test Y[1, :] == y
end