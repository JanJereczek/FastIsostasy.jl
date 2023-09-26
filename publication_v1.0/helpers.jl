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
    t = copy(ds["Time"][:, :])
    lat = copy(ds["Lat"][:, :])
    lon = copy(ds["Lon"][:, :])
    close(ds)
    return Hice, t, lat, lon
end

function vec_dHice(Omega, t, Hitp)
    Hice_vec = [copy(Omega.null) for _ in t]
    for k in eachindex(t)
        for IJ in CartesianIndices(Omega.Lat)
            i, j = Tuple(IJ)
            # println(Omega.Lon[i, j], "  ", Omega.Lat[i, j], "  ", t[k])
            if (150 < Omega.Lon[i, j] < 180)  && (-69 < Omega.Lat[i, j] < -60)
                Hice_vec[k][i, j] = 0.0
            else
                Hice_vec[k][i, j] = Hitp(Omega.Lon[i, j], Omega.Lat[i, j], t[k])
            end
        end
    end
    deltaH = [Hice_vec[k] - Hice_vec[1] for k in eachindex(t)]
    return Hice_vec, deltaH
end