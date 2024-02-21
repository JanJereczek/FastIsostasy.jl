using FastIsostasy, NCDatasets, DelimitedFiles, CairoMakie, JLD2

const DIR = "../data/test4/ICE6G"
const FILE = "newconv-1D-interactivesl=true-bsl=external-N=350"

function load_nc()
    ds = NCDataset("$DIR/$FILE.nc")
    u, ue, b, ssh, maskgrounded, Hice = copy(ds["u"][:, :, :]), copy(ds["ue"][:, :, :]),
        copy(ds["b"][:, :, :]), copy(ds["seasurfaceheight"][:, :, :]),
        copy(ds["maskgrounded"][:, :, :]), copy(ds["Hice"][:, :, :])
    t = copy(ds["t"][:])
    close(ds)
    return t, u, ue, b, ssh, maskgrounded, Hice
end

function load_omega()
    @load "$DIR/$FILE.jld2" fip
    return Float32.(fip.Omega.Lon), Float32.(fip.Omega.Lat)
end

function append3D2nc!(ds, T, Z, var::String)
    ncZ = defVar(ds, var, T, ("lon", "colat", "t"))
    ncZ[:, :, :] = Z
    return nothing
end

function main()
    T = Float32
    colat90 = T.(vec(readdlm("test4/gausslegendre_colatitudes.csv")))
    colat = vcat(colat90, colat90[end] .+ colat90)
    nlat = length(colat)
    nlon = 2*nlat
    dlon = T(360 / nlon)
    lon = 0:dlon:360-dlon
    lon180 = -180:dlon:180-dlon
    lat = vcat(90 .- colat90, -colat90)

    t, u, ue, b, ssh, maskgrounded, Hice = load_nc()
    nt = length(t)
    Lon, Lat = load_omega()
    maxlat = maximum(Lat)
    j1 = findfirst(lat .< maxlat)

    U = zeros(Float32, nlon, nlat, nt)
    B = zeros(Float32, nlon, nlat, nt)
    SSH = zeros(Float32, nlon, nlat, nt)
    M = zeros(Float32, nlon, nlat, nt)
    H = zeros(Float32, nlon, nlat, nt)
    R = zeros(Float32, nlon, nlat, nt)
    for i in 1:nlon, j in j1:nlat
        R = (lon180[i] .- Lon) .^ 2 + (lat[j] .- Lat) .^ 2
        IJ = argmin(R)
        if R[IJ] < 0.5
            U[i, j, :] = u[IJ, :] + ue[IJ, :]
            B[i, j, :] = b[IJ, :]
            SSH[i, j, :] = ssh[IJ, :]
            M[i, j, :] = maskgrounded[IJ, :]
            H[i, j, :] = Hice[IJ, :]
        end
    end

    ds = NCDataset("$DIR/GaussLegendre-$FILE.nc", "c")
    defDim(ds, "lon", length(lon))
    defDim(ds, "colat", length(colat))
    defDim(ds, "t", length(t))
    nclon = defVar(ds, "lon", T, ("lon",))
    nccolat = defVar(ds, "colat", T, ("colat",))
    nct = defVar(ds, "t", T, ("t",))
    nclon[:] = lon
    nccolat[:] = colat
    nct[:] = t
    append3D2nc!(ds, T, U, "u")
    append3D2nc!(ds, T, B, "b")
    append3D2nc!(ds, T, SSH, "ssh")
    append3D2nc!(ds, T, H, "h")
    close(ds)
end

main()