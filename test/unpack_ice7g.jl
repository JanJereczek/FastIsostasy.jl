push!(LOAD_PATH, "../")
using NCDatasets
using JLD2
using FastIsostasy
using CairoMakie

@inline function stereographic_projection(
    lat::T,
    lon::T,
    R::T;
    lat0=T(-71.0),
    lon0=T(0.0),
) where {T<:Real}
    lat, lon, lat0, lon0 = deg2rad.([lat, lon, lat0, lon0])
    k = 2*R / (1 + sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
    x = k * cos(lat) * sin(lon - lon0)
    y = k * (cos(lat0) * sin(lat) - sin(lat0) * cos(lat) * cos(lon-lon0))
    return x, y
end

function main(;make_anim = false)
    prefix = "data/test4/ICE-7G/I7G_NA.VM7_1deg."
    suffix = ".nc"
    tvec = collect(21:-0.5:0)
    H = fill(0f0, 360, 180, length(tvec))
    topo, topo_diff = copy(H), copy(H)
    lat, lon = fill(0f0, 180), fill(0f0, 360)

    for i in eachindex(tvec)
        t = tvec[i]
        if round(t) == t
            t = Int(t)
        end
        filename = string(prefix, t, suffix)
        data = NCDataset(filename, "r") do ds

            if i == 1
                lat .= ds["lat"][:,:]
                lon .= ds["lon"][:,:]
            end

            H[:, :, i] .= ds["stgit"][:,:]
            topo[:, :, i] .= ds["Topo"][:,:]
            topo_diff[:, :, i] .= ds["Topo_Diff"][:,:]
        end
    end

    LAT, LON = meshgrid(lat, lon)
    R = 6371f3
    X, Y = fill(0f0, size(LAT)), fill(0f0, size(LAT))
    for i in axes(LAT, 1), j in axes(LAT, 2)
        x, y = stereographic_projection(LAT[i,j], LON[i,j], R)
        X[i,j] = x
        Y[i,j] = y
    end
    xsp, ysp = stereographic_projection(-90f0, 0f0, R)
    X .-= xsp
    Y .-= ysp

    n = 8
    L = 3000e3
    xcartesian = range(-L, stop = L, length = 2^n)
    ycartesian = copy(xcartesian)
    Xcartesian, Ycartesian = meshgrid(xcartesian, ycartesian)
    Hcartesian = fill(0f0, size(Xcartesian)..., length(tvec))
    for i in axes(Xcartesian, 1), j in axes(Xcartesian, 2)
        kmin = argmin( (X .- Xcartesian[i,j]).^2 + (Y .- Ycartesian[i,j]).^2 )
        Hcartesian[i, j, :] = H[kmin, :]
    end

    jldsave(
        "data/test4/ice7g.jld2",
        tvec = tvec,
        lat = lat,
        lon = lon,
        H = H,
        Xcartesian = Xcartesian,
        Ycartesian = Ycartesian,
        Hcartesian = Hcartesian,
        topo = topo,
        topo_diff = topo_diff,
    )

    if make_anim
        idx = Observable(1)
        H = @lift(Hcartesian[:, :, $idx]')
        cmap = :rainbow # :ice
        clim = (1e-8, 4500)
        fig = Figure(resolution = (800, 800))
        ax = Axis(fig[1, 1], aspect = DataAspect())
        hm = heatmap!(
            ax,
            Xcartesian,
            Ycartesian,
            H,
            colormap = cmap,
            colorrange = clim,
            lowclip = :white,
            highclip = :black,
        )
        Colorbar(fig[1,2], hm, height = Relative(0.7))

        record(fig, "ice_load_history.mp4", axes(Hcartesian, 3); framerate = 10) do i
            idx[] = i
        end
    end
end

main()