# using Proj

# @testset "Proj.jl" begin
#     lon, lat = 67.8, -85.4
#     lat_ts = -71.0
#     forwardproj = Proj.Transformation("EPSG:4326",
#         "+proj=stere +lat_0=-90 +lat_ts=$lat_ts", always_xy=true)
#     x, y = forwardproj(lon, lat)
#     lon_, lat_ = inv(forwardproj)(x, y)
#     @test lon_ ≈ lon
#     @test lat_ ≈ lat
# end

@testset "ij indexing for xy" begin
    x, y = 1.0:10, 1.0:5
    X, Y = meshgrid(collect(x), collect(y))
    @test X[:, 1] == x
    @test Y[1, :] == y
end