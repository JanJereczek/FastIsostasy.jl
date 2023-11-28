function unpack(results::FastIsoProblem)
    Omega, p = reinit_structs_cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end

latexify(x) = [L"%$xi $\,$" for xi in x]
latexticks(x) = (x, latexify(x))
diagslice(X, N2, N4) = diag(X)[N2:N2+N4]

janjet = [:gray10, :cornflowerblue, :orange, :red3]
janjet_small = [:purple4, :royalblue, :cornflowerblue, :orange, :red3]

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

function indices_latychev2023_indices(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]
    return idx, x
end

function load_latychev_gaussian(dir::String, idx)
    files = readdir(dir)
    # nr = size( readdlm(joinpath(dir, files[1]), ','), 1 )
    nr = length(files)
    u = zeros(sum(idx), nr)
    for i in eachindex(files)
        u[:, i] = readdlm(joinpath(dir, files[i]), ',')[idx, 2]
    end
    return u
end




















function load_bsl()
    file = "../data/ICE6Gzip/IceT.I6F_C.131QB_VM5a_1deg.nc"
    ds = NCDataset("$file", "r")
    lon_360, lat, tpaleo = copy(ds["Lon"][:]), copy(ds["Lat"][:]), copy(ds["Time"][:])
    Hice_360 = copy(ds["stgit"][:, :, :])
    close(ds)
    lon, Hice = lon360tolon180(lon_360, Hice_360)
    t = -tpaleo .* 1e3

    z = interp_etopo(lat, lon)
    Hice_af = [haf(Hice[:, :, k], z) for k in eachindex(t)]
    cellsurface = get_cellsurface(lat, lon)
    dV_af = [sum((Hice_af[k] - Hice_af[1]) .* cellsurface) for k in eachindex(t)]
    Ao = 3.625e14
    sl = -dV_af ./ Ao
    return t, sl, linear_interpolation(t, sl, extrapolation_bc = Flat())
end


function interp_etopo(lat, lon)
    file = "../data/bathymetry/ETOPO_2022_v1_60s_N90W180_bed.nc"
    ds = NCDataset("$file", "r")
    lon_etopo, lat_etopo = copy(ds["lon"][:]), copy(ds["lat"][:])
    z = copy(ds["z"][:, :])
    close(ds)
    itp = linear_interpolation((lon_etopo, lat_etopo), z)
    Lon, Lat = meshgrid(lon, lat)
    return itp.(Lon, Lat)
end
haf(H, b) = H + min.(b, 0) .* (1028 / 910)

function get_cellsurface(lat::Vector{T}, lon::Vector{T}) where {T<:AbstractFloat}
    R = 6371e3                     # Earth radius at equator (m)
    k = 1 ./ cos.( deg2rad.(lat) )
    dphi = mean([mean(diff(lat)), mean(diff(lon))])
    meridionallength_cell = deg2rad(dphi) * R
    azimutallength_cell = meridionallength_cell ./ k
    cellsurface = fill(meridionallength_cell, length(lon)) * azimutallength_cell'
    return cellsurface
end
