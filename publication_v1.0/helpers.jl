function unpack(results::FastIsoProblem)
    Omega, p = reinit_structs_cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end

not(x::Bool) = !x
latexify(x) = [L"%$xi $\,$" for xi in x]
latexticks(x) = (x, latexify(x))
diagslice(X, N2, N4) = diag(X)[N2:N2+N4]

janjet = [:gray10, :cornflowerblue, :orange, :red3]
janjet_small = [:purple4, :royalblue, :cornflowerblue, :orange, :red3]

function lon360tolon180(lon, X)
    permidx = lon .> 180
    lon180 = vcat(lon[permidx] .- 360, lon[not.(permidx)])
    X180 = cat(X[permidx, :, :], X[not.(permidx), :, :], dims=1)
    return lon180, X180
end

function load_ice6g()
    file = "../data/ICE6Gzip/IceT.I6F_C.131QB_VM5a_1deg.nc"
    ds = NCDataset(file)
    Hice = copy(ds["stgit"][:, :])
    t, lat, lon = copy(ds["Time"][:, :]), copy(ds["Lat"][:, :]), copy(ds["Lon"][:, :])
    close(ds)
    return Hice, t, lat, lon
end

function load_ice6gd(; key = "IceT")
    file = "../data/ICE6G_D/ICE6GD_$key.nc"
    ds = NCDataset(file)
    t, lon, lat = copy(ds["Time"][:, :]), copy(ds["Lon"][:, :]), copy(ds["Lat"][:, :])
    Hice = copy(ds["$key"])
    close(ds)

    t .*= -1
    lon180, Hice180 = lon360tolon180(lon, Hice)
    Hice_itp = linear_interpolation((lon180, lat, t), Hice180, extrapolation_bc = Flat())

    return t, lat, lon, Hice, Hice_itp
end

function vec_dHice(Omega, Lon, Lat, t, Hitp)
    Hice_vec = [copy(Array(Omega.null)) for _ in t]
    for k in eachindex(t)
        for IJ in CartesianIndices(Lat)
            i, j = Tuple(IJ)
            # println(Omega.Lon[i, j], "  ", Omega.Lat[i, j], "  ", t[k])
            if (150 < Lon[i, j] < 180)  && (-69 < Lat[i, j] < -60)
                Hice_vec[k][i, j] = 0.0
            else
                Hice_vec[k][i, j] = Hitp(Lon[i, j], Lat[i, j], t[k])
            end
        end
    end
    deltaH = [Hice_vec[k] - Hice_vec[1] for k in eachindex(t)]
    return Hice_vec, deltaH
end