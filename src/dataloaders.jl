const isos_data = "https://github.com/JanJereczek/isostasy_data"

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
    ############################## Masks ######################################
    if name == "Antarctic3RegionMask"
        return load_antarctic_3regionmask()
    ########################## Param and forcing fields #######################
    elseif name == "OceanSurfaceFunctionETOPO2022"
        return load_oceansurface_data(; kwargs...)
    elseif name == "BedMachine3"
        return load_bedmachine3(; kwargs...)
    elseif name == "ICE6G_D"
        return load_ice6gd(; kwargs...)
    elseif name == "Wiens2022"
        return load_wiens2022(; kwargs...)
    elseif name == "Lithothickness_Pan2022"
        return load_lithothickness_pan2022(; kwargs...)
    elseif name == "Viscosity_Pan2022"
        return load_logvisc_pan2022(; kwargs...)
    ########################## Model outputs ##################################
    elseif name == "Spada2011"
        return load_spada2011(; kwargs...)
    # elseif name == "LatychevGaussian"
    #     return load_latychev_gaussian(dir::String, x_lb::Real, x_ub::Real)
    elseif name == "LatychevICE6G"
        return load_latychev2023_ICE6G(; kwargs...)
    end
end



#############################################################
# Mask
#############################################################

function load_antarctic_3regionmask()
    link = "$isos_data/raw/main/tools/masks/ANT-16KM_BASINS-nasa.nc"
    tmp = download(link, tempdir() *"/"* basename(link))
    x = ncread(tmp, "xc")
    y = ncread(tmp, "yc")
    mask = ncread(tmp, "mask_regions")

    mask_itp = linear_interpolation((x, y), mask, extrapolation_bc = Flat())
    println("returning: (x, y), mask, interpolator")
    return (x, y), mask, mask_itp
end

#############################################################
# Parameter fields
#############################################################
function load_oceansurface_data(; T = Float64, verbose = true)
    link = "$isos_data/raw/main/tools/ocean_surface/dz=0.1m.txt"
    tmp = download(link)
    data = readdlm(tmp)
    z, A = T.(data[:, 1]), T.(data[:, 2])
    return z, A, nothing
end

function load_bedmachine3(; var = "bed", T = Float64)
    link = "$isos_data/raw/main/topography/BedMachineAntarctica-v3-sparse.nc"
    tmp = download(link, tempdir() *"/"* basename(link))
    var = T.(ncread(tmp, "$var"))
    x, y = T.(ncread(tmp, "x")), T.(ncread(tmp, "y"))
    itp = linear_interpolation((x, reverse(y)), reverse(var, dims=2))
    println("returning: (x, y), var, interpolator")
    return (x, y), var, itp
end

function load_ice6gd(; var = "IceT")
    link = "$isos_data/raw/main/ice_history/ICE6G_D/ICE6GD_$var.nc"
    tmp = download(link, tempdir() *"/"* basename(link))
    t, lon, lat = ncread(tmp, "Time"), ncread(tmp, "Lon"), ncread(tmp, "Lat")
    X = ncread(tmp, "$var")

    t .*= -1
    lon180, X180 = lon360tolon180(lon, X)
    Hice_itp = linear_interpolation((lon180, lat, t), X180, extrapolation_bc = 0.0)

    println("returning (var determined by kwarg): (lon180, lat, t), var, interpolator")
    return (lon180, lat, t), X180, Hice_itp # check if this should be X or X180
end

function load_wiens2022(; extrapolation_bc = Throw())
    link = "$isos_data/raw/main/earth_structure/viscosity/wiens2022.nc"
    tmp = download(link, tempdir() *"/"* basename(link))
    x, y, z = ncread(tmp, "x"), ncread(tmp, "y"), ncread(tmp, "z")
    log10visc = ncread(tmp, "log10visc")
    println("returning: (x, y, z), log10visc, interpolator")
    return (x, y, z), log10visc, linear_interpolation((x, y, z), log10visc,
        extrapolation_bc = extrapolation_bc)
end

function load_lithothickness_pan2022()
    link = "$isos_data/raw/main/earth_structure/lithothickness/pan2022.llz"
    tmp = download(link, tempdir() *"/"* basename(link))
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
    link = "$isos_data/raw/main/earth_structure/viscosity/pan2022.nc"
    tmp = download(link, tempdir() *"/"* basename(link))
    lon = ncread(tmp, "lon")
    lat = ncread(tmp, "lat")
    r = ncread(tmp, "r")
    logvisc = ncread(tmp, "eta")
    logvisc_itp = linear_interpolation((lon, lat, r), logvisc, extrapolation_bc = Flat())
    println("returning: (lon180, lat, r), eta (in log10 space), interpolator")
    return (lon, lat, r), logvisc, logvisc_itp
end

# Preliminary Reference Earth Model
"""
    load_prem()

Load Preliminary Reference Earth Model (PREM) from Dzewonski and Anderson (1981).
"""
function load_prem()
    # radius, depth, density, Vpv, Vph, Vsv, Vsh, eta, Q-mu, Q-kappa
    M = readdlm(joinpath(@__DIR__, "input/PREM_1s.csv"), ',')[:, 1:7]
    M .*= 1e3
    return ReferenceSolidEarthModel([M[:, j] for j in axes(M, 2)]...)
end


#############################################################
# Model output
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
    link = "$isos_data/raw/main/model_outputs/Spada-2011/$case.txt"
    tmp = download(link, tempdir() *"/"* basename(link))
    X = readdlm(tmp)
    if occursin("n_", case)
        reverse!(X, dims = 1)
    end
    return (theta, t), X, linear_interpolation((theta, t), X)
end

function load_latychev_test3(; case = "E0L1V1")
    link = "$isos_data/raw/main/model_outputs/SwierczekLatychev-2023/test3/$case.nc"
    tmp = download(link, tempdir() *"/"* basename(link))
    r = ncread(tmp, "r")
    t = ncread(tmp, "t")
    u = ncread(tmp, "u")
    println("returning: (r, t), u, interpolator")
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
    # X = Array{Float64, 3}(undef, nlon, nlat, length(tlaty))
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
    file_is_remote = true,
    remote_path = "$isos_data/raw/main/tools/green_elastic/elasticgreencoeffs_farrell1972.txt",
    local_path = nothing)

    if file_is_remote
        tmp = download(remote_path, tempdir() *"/"* basename(remote_path))
    else
        tmp = local_path
    end
    data = readdlm(tmp)

    # rm is column 1 converted to meters (and some extra factor)
    # GE /(10^12 rm) is vertical displacement in meters (applied load is 1kg)
    # GE corresponds to column 2
    return T.(data[:, 2]), T.(data[:, 3])
end


"""
    get_greenintegrand_coeffs(T)

Return the load response coefficients with type `T`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function load_viscous_kelvin_function(T::Type)
    remote_path = "$isos_data/raw/main/tools/green_viscous/tabulated_kelvin_function_elra.txt"
    tmp = download(remote_path, tempdir() *"/"* basename(remote_path))
    data = T.(readdlm(tmp, ','))
    return data[:, 1], data[:, 2]   # rn, kei
end

