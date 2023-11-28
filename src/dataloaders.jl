"""
    load_dataset(name) → (dims), field, interpolator

Return the `dims::Tuple{Vararg{Vector}}`, the `field<:Array` and the `interpolator`
corresponding to a data set defined by a unique `name::String`. For instance:

```julia
(lon180, lat, t), Hice, Hice_itp = load_dataset("ICE6G_D")
```

Following options are available for parameter fields:
 - "ICE6G_D": ice loading history from ICE6G_D.
 - "Wiens2022": viscosity field from (Wiens et al. 2022)
 - "Lithothickness_Pan2022": lithospheric thickness field from (Pan et al. 2022)
 - "Viscosity_Pan2022": viscosity field from (Pan et al. 2022)

Following options are available for model results:
 - "Spada2011"
 - "LatychevGaussian"
 - "LatychevICE6G"
"""
function load_dataset(name::String; kwargs...)
    ########################## Param and forcing fields #######################
    if name == "OceanSurfaceFunctionETOPO2022"
        return load_oceansurfacefunction(; kwargs...)
    elseif name == "BedMachine3"
        return load_bedmachine3(; kwargs...)
    elseif name == "ICE6G_D"
        return load_ice6gd(; kwargs...)
    elseif name == "Wiens2022"
        return load_wiens2022(; kwargs...)
    elseif name == "Lithothickness_Pan2022"
        return load_lithothickness_pan2022(; kwargs...)
    elseif name == "Viscosity_Pan2022"
        return load_3Dvisc_pan2022(; kwargs...)
    ########################## Model outputs ##################################
    elseif name == "Spada2011"
        return load_spada2011(; kwargs...)
    # elseif name == "LatychevGaussian"
    #     return load_latychev_gaussian(dir::String, x_lb::Real, x_ub::Real)
    elseif name == "LatychevICE6G"
        return load_laty_ICE6G(; kwargs...)
    end
end

#############################################################
# Parameter fields
#############################################################
function load_oceansurfacefunction()
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/ocean_surface/dz=0.1m.jld2"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    @load "$tmp" z_support A_support
    z, A = collect(z_support), collect(A_support)
    itp = linear_interpolation(z, A)    # , extrapolation_bc = Line()
    return z, A, itp
end

function load_bedmachine3(; var = "bed", T = Float64)
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/topography/"*
        "BedMachineAntarctica-v3-sparse.nc"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    ds = NCDataset(tmp, "r")
    var = T.(ds["$var"][:, :])
    x, y = T.(ds["x"][:]), T.(ds["y"][:])
    close(ds)
    itp = linear_interpolation((x, reverse(y)), reverse(var, dims=2))
    return (x, y), var, itp
end

function bathymetry(Omega::ComputationDomain)
    T = Float32
    ds = NCDataset(joinpath(@__DIR__, "../data/bathymetry/ETOPO_2022_v1_60s_N90W180_bed.nc"),"r")
    lon, lat = T.(ds["lon"][:]), T.(ds["lat"][:])
    z = T.(ds["z"][:, :])
    close(ds)
    itp = linear_interpolation((lon, lat), z, extrapolation_bc = Flat())
    return itp.(Array(Omega.Lon), Array(Omega.Lat))
end

function load_ice6gd(; var = "IceT")
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/ice_history/"*
        "ICE6G_D/ICE6GD_$var.nc"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    ds = NCDataset(tmp, "r")
    t, lon, lat = copy(ds["Time"][:]), copy(ds["Lon"][:]), copy(ds["Lat"][:])
    Hice = copy(ds["$var"][:, :, :])
    close(ds)

    t .*= -1
    lon180, Hice180 = lon360tolon180(lon, Hice)
    Hice_itp = linear_interpolation((lon180, lat, t), Hice180, extrapolation_bc = 0.0)

    println("returning: (lon180, lat, t), Hice, interpolator")
    return (lon180, lat, t), Hice, Hice_itp
end

function load_wiens2022(; extrapolation_bc = Throw())
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/parameter_fields/"*
        "viscosity/wiens2022.nc"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    ds = NCDataset(tmp, "r")
    x, y, z = copy(ds["x"][:]), copy(ds["y"][:]), copy(ds["z"][:])
    log10visc = copy(ds["log10visc"][:, :, :])
    close(ds)
    println("returning: (x, y, z), log10visc, interpolator")
    return (x, y, z), log10visc, linear_interpolation((x, y, z), log10visc,
        extrapolation_bc = extrapolation_bc)
end

function load_lithothickness_pan2022()
    link = "https://raw.githubusercontent.com/JanJereczek/IsostasyData/main/"*
        "parameter_fields/lithothickness/pan2022.llz"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    data, head = readdlm(tmp, header = true)
    Lon_vec, Lat_vec, T_vec = data[:, 1], data[:, 2], data[:, 3]
    lon, lat = unique(Lon_vec), unique(Lat_vec)
    nlon, nlat = length(lon), length(lat)
    Tlitho = reshape(T_vec, nlon, nlat)
    reverse!(Tlitho, dims=2)
    reverse!(lat)
    lon180, Tlitho180 = lon360tolon180(lon, Tlitho)
    itp = linear_interpolation((lon180, lat), Tlitho180[:, :, 1], extrapolation_bc = Flat())

    println("returning: (lon180, lat), Tlitho, interpolator")
    return (lon180, lat), Tlitho, itp
end

function load_logvisc_pan2022()
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/parameter_fields/viscosity/pan2022.nc"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    ds = NCDataset(tmp, "r")
    lon = copy(ds["lon"][:])
    lat = copy(ds["lat"][:])
    r = copy(ds["r"][:])
    logvisc = copy(ds["eta"][:, :, :])
    close(ds)
    logvisc_itp = linear_interpolation((lon, lat, r), logvisc, extrapolation_bc = Flat())
    println("returning: (lon180, lat, r), eta (in log10 space), interpolator")
    return (lon, lat, r), logvisc, logvisc_itp
end

#############################################################
# Model outputs
#############################################################

function spada_dims()
    theta = 0:0.1:20
    t = [0, 1, 2, 5, 10, 100] .* 1e3
    return theta, t
end

spada_cases() = ["u_cap", "u_disc", "dudt_cap", "dudt_disc", "n_cap", "n_disc"]

function load_spada2011()
    theta = Dict{String, Vector{Float64}}()
    t = Dict{String, Vector{Float64}}()
    X = Dict{String, Matrix{Float64}}()
    Xitp = Dict()
    for case in spada_cases()
        (i, j), k, l = load_spada2011(case)
        theta[case] = i
        t[case] = j
        X[case] = k
        Xitp[case] = l
    end
    println("returning dictionnaries (keys = spada_cases): (theta, t), X, interpolator")
    return (theta, t), X, Xitp
end

function load_spada2011(case)
    theta, t = spada_dims()
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/model_outputs/"*
        "Spada-2011/$case.jld2"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    @load "$tmp" X
    if occursin("n_", case)
        reverse!(X, dims = 1)
    end
    return (theta, t), X, linear_interpolation((theta, t), X)
end

function load_latychev_test3(; case = "E0L1V1")
    link = "https://github.com/JanJereczek/IsostasyData/raw/main/model_outputs/"*
        "SwierczekLatychev-2023/test3/$case.nc"
    tmp = Downloads.download(link, tempdir() *"/"* basename(link))
    ds = NCDataset(tmp, "r")
    r = copy(ds["r"][:])
    t = copy(ds["t"][:])
    u = copy(ds["u"][:, :])
    close(ds)
    return (r, t), u, linear_interpolation((r, t), u)
end

function load_latychev2023_ICE6G(; case = "1D", var = "R")

    if case == "1D"
        latydir = joinpath(@__DIR__, "../data/Latychev/ICE6G/dense/1D/$var")
    elseif case == "3D"
        latydir = joinpath(@__DIR__, "../data/Latychev/ICE6G/dense/3D/$var")
    end
    timestep_file = joinpath(@__DIR__, "../data/Latychev/ICE6G/dense/tt_25.dat")
    tlaty = vec(readdlm(timestep_file))
    latyfiles = readdir(latydir)

    latyfiles = latyfiles[1:end-1]
    tlaty = tlaty[1:end-1]
    nlon, nlat = 512, 256
    X = zeros(nlon, nlat, length(tlaty))

    for k in eachindex(latyfiles)
        X[:, :, k] = reshape(vec(readdlm(joinpath(latydir, latyfiles[k]))),
            nlon, nlat)
    end

    gllatlon_file = joinpath(@__DIR__, "../data/Latychev/ICE6G/gl256_LatLon")
    LonLat, header = readdlm(gllatlon_file, header = true)
    Lon = reshape(LonLat[:, 1], nlon, nlat)
    Lat = reshape(LonLat[:, 2], nlon, nlat)
    Lon, Lat = reverse(Lon, dims = 2), reverse(Lat, dims = 2)
    X = reverse(X, dims = 2)

    Lon, X = lon360tolon180(Lon[:, 1], X)
    lon, lat = Lon[:, 1], Lat[1, :]
    itp = linear_interpolation((lon, lat, tlaty), X, extrapolation_bc = Flat())

    println("returning (var determined by kwarg): (lon, lat, t), var, interpolator")
    return (lon, lat, tlaty), X, itp
end

#############################################################
# Green coefficients for elastic displacement
#############################################################

"""
    get_greenintegrand_coeffs(T)

Return the load response coefficients with type `T`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function get_greenintegrand_coeffs(T::Type;
    file = joinpath(@__DIR__, "input/elasticgreencoeffs_farrell1972.jld2"))
    data = jldopen(file)
    # rm is column 1 converted to meters (and some extra factor)
    # GE /(10^12 rm) is vertical displacement in meters (applied load is 1kg)
    # GE corresponds to column 2
    return T.(data["rm"]), T.(data["GE"])
end

#############################################################
# Preliminary Reference Earth Model
#############################################################

"""
    load_prem()

Load Preliminary Reference Earth Model (PREM) from Dzewonski and Anderson (1981).
"""
function load_prem()
    # radius, depth, density, Vpv, Vph, Vsv, Vsh, eta, Q-mu, Q-kappa
    M = readdlm(joinpath(@__DIR__, "input/PREM_1s.csv"), ',')[:, 1:7]
    M .*= 1e3
    return ReferenceEarthModel([M[:, j] for j in axes(M, 2)]...)
end