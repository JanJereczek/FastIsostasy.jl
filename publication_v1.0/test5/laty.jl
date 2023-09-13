function load_laty_ICE6G()
    latydir = "../data/Latychev/ICE6G/r1d_ICE6G"
    latyfiles = readdir(latydir)
    tlaty = [-120, -26, -21, -16, -11, -6, -2, -1, -0.125, 0, 0.125] .* 1e3

    latyfiles = latyfiles[1:end-1]
    tlaty = tlaty[1:end-1]
    nlon, nlat = 512, 256
    ulaty = zeros(nlon, nlat, length(tlaty))

    for k in eachindex(latyfiles)
        ulaty[:, :, k] = reshape(vec(readdlm(joinpath(latydir, latyfiles[k]))), nlon, nlat)
    end

    gllatlon_file = "../data/Latychev/ICE6G/gl256_LatLon"
    LonLat, header = readdlm(gllatlon_file, header = true)
    Lon = reshape(LonLat[:, 1], nlon, nlat)
    Lat = reshape(LonLat[:, 2], nlon, nlat)
    Lon, Lat = reverse(Lon, dims = 2), reverse(Lat, dims = 2)
    ulaty = reverse(ulaty, dims = 2)

    Lon, ulaty = lon360tolon180(Lon[:, 1], ulaty)

    for k in eachindex(tlaty)
        fig = heatmap(ulaty[:, :, k], colorrange = (-500, 500), colormap = :PuOr)
        save("plots/test5/laty-ICE6G-global$(tlaty[k]).png", fig)
    end
    itp = linear_interpolation((Lon[:, 1], Lat[1, :], tlaty), ulaty,
        extrapolation_bc = Flat())

    return tlaty, ulaty, Lon, Lat, itp
end