
function load_dataset(name::String)
    if name == "Etopo2022"
        return load_bathymetry()
    elseif name == "Spada2011"
        return load_spada2011()
    elseif name == "LatychevGaussian"
        return load_latychev_gaussian(dir::String, x_lb::Real, x_ub::Real)
    elseif name == "LatychevICE6G-1D"
        return load_laty_ICE6G(case = "1D")
    elseif name == "LatychevICE6G-3D"
        return load_laty_ICE6G(case = "3D")
    elseif name == "ICE6G_C"
        return 
    elseif name == "ICE6G_D"
    elseif name == "Wiens_2022"

    elseif name == "Ivins_2022"
        return nothing
    end
end

"""
    load_bathymetry()

Load the bathymetry map from ETOPO-2022 on 1 minute arclength resolution.
"""
function load_bathymetry()
    # https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/60s/60s_bed_elev_netcdf/catalog.html?dataset=globalDatasetScan/ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc
    filename = joinpath(@__DIR__, "data/ETOPO_2022_v1_60s_N90W180_bed.nc")
    ds = NCDataset(filename, "r")
    lat = copy(ds["lat"][:,:])
    lon = copy(ds["lon"][:,:])
    bedrock_missing = copy(ds["z"][:,:])
    close(ds)

    bedrock = zeros(size(bedrock_missing))
    bedrock .= bedrock_missing

    return lon, lat, bedrock
end


function load_laty_3Dvisc()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Latychev/ICE6G/dense/stp_model_MH"
    data = vec(readdlm(file, skipstart = 2))
    r = data[1:64]
    eta_ratio_vec = data[65:end]

    lon = collect(range(0, stop = 360, length = 721))
    lat = collect(range(90, stop = -90, length = 361))
    nr, nlat, nlon = length(r), length(lat), length(lon)
    eta_ratio = reshape(eta_ratio_vec, (nlon, nlat, nr))
    heatmap(eta_ratio[:, :, end])
    reverse!(eta_ratio, dims = 2)
    reverse!(lat)
    heatmap(eta_ratio[:, :, end])
    eta1D = [3.16, 3.16, 1.58, 1.58, 0.5, 0.5] .* 1e21
    r_eta1D = [3480000.0, 5173800.9, 5173800.95, 5701000.0, 5701000.05, 6371000.0]
    eta1D_itp = linear_interpolation(r_eta1D, eta1D)
    eta1D_3D = zeros((nlon, nlat, nr))
    for i in axes(eta1D_3D, 1), j in axes(eta1D_3D, 2)
        eta1D_3D[i, j, :] .= eta1D_itp.(r)
    end
    eta = eta1D_3D .* (10 .^ eta_ratio)
    lon180, eta180 = lon360tolon180(lon, eta)
    heatmap(eta180[:, :, end-5])
    eta_itp = linear_interpolation((lon180, lat, r), eta180, extrapolation_bc = Flat())
    return eta_itp
end

function load_litho_thickness_laty()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Latychev/ICE6G/dense/LAB.llz"
    data, head = readdlm(file, header = true)
    Lon_vec, Lat_vec, T_vec = data[:, 1], data[:, 2], data[:, 3]
    lon, lat = unique(Lon_vec), unique(Lat_vec)
    nlon, nlat = length(lon), length(lat)
    Lon = reshape(Lon_vec, nlon, nlat)
    Lat = reshape(Lat_vec, nlon, nlat)
    Tlitho = reshape(T_vec, nlon, nlat)
    reverse!(Tlitho, dims=2)
    reverse!(Lat, dims=2)
    reverse!(lat)
    heatmap(Tlitho)
    lon180, Tlitho180 = lon360tolon180(lon, Tlitho)
    itp = linear_interpolation((lon180, lat), Tlitho180[:, :, 1], extrapolation_bc = Flat())
    return itp
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

#############################################################
# Model outputs
#############################################################

function load_spada2011()
    prefix ="../testdata/Spada/"
    cases = ["u_cap", "u_disc", "dudt_cap", "dudt_disc", "n_cap", "n_disc"]
    snapshots = ["0", "1", "2", "5", "10", "inf"]
    data = Dict{String, Vector{Matrix{Float64}}}()
    for case in cases
        tmp = Matrix{Float64}[]
        for snapshot in snapshots
            fname = string(prefix, case, "_", snapshot, ".csv")
            append!(tmp, [readdlm(fname, ',', Float64)])
        end
        data[case] = tmp
    end
    return data
end

function load_latychev_gaussian(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]

    u = zeros(length(x), length(files))
    for i in eachindex(files)
        file = files[i]
        # println( file, typeof( readdlm(joinpath(dir, file), ',')[:, 1] ) )
        u[:, i] = readdlm(joinpath(dir, file), ',')[idx, 2]
    end
    # u .-= u[:, 1]

    return x, u
end

function load_laty_ICE6G(; case = "1D")
    if case == "1D_sparse"
        latydir = "../data/Latychev/ICE6G/r1d_ICE6G"
        tlaty = [-120, -26, -21, -16, -11, -6, -2, -1, -0.125, 0, 0.125] .* 1e3
    elseif case == "1D"
        latydir = "../data/Latychev/ICE6G/dense/1D/R"
        tlaty = vec(readdlm("../data/Latychev/ICE6G/dense/tt_25.dat"))
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

#############################################################
# Parameter fields
#############################################################

"""
    load_wiens2021(Omega)

Load the viscosity layers estimated in [^Wiens2021] into a 3D array with \$ x, y, z \$
respectively the first, second and third dimension of the array.

Will be replaced in future by `load_paramfield(publication = "Wiens2021")`.
"""
function load_wiens2021(Omega::ComputationDomain{T, M}; halfspace_logvisc::Real = 21) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}

    jld2file = "Wiens2021_Nx=64_Ny=64.jld2"
    dir = joinpath(@__DIR__, "../testdata/Wiens")
    # if jld2file in readdir(dir)
    #     println("Preprocessed JLD2 file already exists. Skipping pre-processing.")
    # else
    #     jld2_wiens2021(Omega, jld2file, dir)
    # end
    @load "$dir/$jld2file" logvisc3D logvisc_interpolators

    lv = 10.0 .^ cat( [itp.(Omega.X, Omega.Y) for itp in logvisc_interpolators]...,
        fill(T(halfspace_logvisc), Omega.Nx, Omega.Ny), dims=3)
    return lv
end

function jld2_wiens2021(Omega::ComputationDomain, jld2file::String, dir::String)

    X, Y, Nx, Ny = Omega.X, Omega.Y, Omega.Nx, Omega.Ny
    x, y = X[:, 1], Y[1, :]
    rawdata = [readdlm(file) for file in readdir(dir, join = true)]
    logvisc = [wiens_filter_nan_viscosity(M) for M in rawdata]
    km2m!(logvisc)

    z = [100e3, 200e3, 300e3]
    logvisc3D = zeros(Float64, (Nx, Ny, length(z)))
    for k in axes(logvisc3D, 3)
        for i in axes(logvisc3D, 1), j in axes(logvisc3D, 2)
            logvisc3D[i, j, k] = wiens_get_closest_eta(X[i,j], Y[i,j], logvisc[k])
        end
    end

    logvisc_interpolators = [linear_interpolation( (x, y), logvisc3D[:, :, k],
        extrapolation_bc = Flat() ) for k in axes(logvisc3D, 3)]
    jldsave(joinpath(dir, jld2file), logvisc3D = logvisc3D,
        logvisc_interpolators = logvisc_interpolators)
    return nothing
end

function wiens_filter_nan_viscosity(M::Matrix{T}) where {T<:AbstractFloat}
    return M[.!isnan.(M[:, 3]), :]
end

function wiens_get_closest_eta(x::T, y::T, M::Matrix{T}) where {T<:AbstractFloat}
    l = argmin( (x .- M[:, 1]) .^ 2 + (y .- M[:, 2]) .^ 2 )
    return M[l, 3]
end

function km2m!(V::Vector{Matrix{T}}) where {T<:AbstractFloat}
    for i in eachindex(V)
        for j in [1, 2, 4]
            V[i][:, j] .*= T(1e3) 
        end
    end
end