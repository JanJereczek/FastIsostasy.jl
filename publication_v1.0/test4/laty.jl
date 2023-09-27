function load_laty_ICE6G(; case = "1D")
    if case == "1D"
        latydir = "../data/Latychev/ICE6G/r1d_ICE6G"
        tlaty = [-120, -26, -21, -16, -11, -6, -2, -1, -0.125, 0, 0.125] .* 1e3
    elseif case == "3D"
        latydir = "../data/Latychev/ICE6G/dense/3D/R"
        tlaty = vec(readdlm("../data/Latychev/ICE6G/dense/tt_25.dat"))
    end
    latyfiles = readdir(latydir)

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
        fig = Figure(resolution = (1600, 900), fontsize = 30)
        ax = Axis(fig[1,1], aspect = DataAspect())
        hidedecorations!(ax)
        hm = heatmap!(ulaty[:, :, k], colorrange = (-400, 400), colormap = :PuOr)
        Colorbar(fig[1, 2], hm, height = Relative(0.8), label = "Displacement (m)")
        save("plots/test4/laty/ICE6G-$case-global-lorange-$(tlaty[k]).png", fig)
    end
    itp = linear_interpolation((Lon[:, 1], Lat[1, :], tlaty), ulaty,
        extrapolation_bc = Flat())

    return tlaty, ulaty, Lon, Lat, itp
end