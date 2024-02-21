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
    return t, u + ue, b, ssh, Hice
end

function argsmallest(A::AbstractArray{T,N}, n::Integer) where {T,N}
    # should someone ask more elements than array size, just sort array

    if n>= length(vec(A))
      ind=collect(1:length(vec(A)))
      ind=sortperm(A[ind])
      return CartesianIndices(A)[ind]
    end
    # otherwise 
    ind=collect(1:n)
    mymax=maximum(A[ind])
    for j=n+1:length(vec(A))
    if A[j]<mymax
        getout=findmax(A[ind])[2]
        ind[getout]=j
        mymax=maximum(A[ind])
    end
    end
    ind=ind[sortperm(A[ind])]
    
    return CartesianIndices(A)[ind]
end

function load_omega()
    @load "$DIR/$FILE.jld2" fip
    return Float32.(fip.Omega.Lon), Float32.(fip.Omega.Lat)
end

function save2txt(filename, ks)
    ds = NCDataset(filename)
    u, b, ssh, h = copy(ds["u"][:, :, ks]), copy(ds["b"][:, :, ks]),
        copy(ds["ssh"][:, :, ks]), copy(ds["h"][:, :, ks])
    lon, colat, t = copy(ds["lon"][:]), copy(ds["colat"][:]), copy(ds["t"][ks])
    Lon, Colat = meshgrid(lon, colat)
    for k in eachindex(t)
        data = hcat(vec(Lon), vec(Colat), vec(u[:, :, k]), vec(b[:, :, k]), vec(ssh[:, :, k]),
            vec(h[:, :, k]))
        open("t=$(t[k]).txt", "w") do io
            writedlm(io, data)
        end
    end
    close(ds)
end

function append3D2nc!(ds, T, Z, var::String)
    ncZ = defVar(ds, var, T, ("lon", "colat", "t"))
    ncZ[:, :, :] = Z[:, :, :]
    return nothing
end

function main()
    T = Float32
    colat90 = T.(vec(readdlm("test4/gausslegendre_colatitudes.csv")))
    lat = vcat( 90 .- colat90, reverse(colat90) .- 90 )
    colat = 90 .- lat

    nlat = length(colat)
    nlon = 2*nlat
    dlon = T(360 / nlon)
    lon = 0:dlon:360-dlon

    t, u, b, ssh, Hice = load_nc()
    nt = length(t)
    Lon, Lat = load_omega()
    Lon[Lon .< 0] = 360 .+ Lon[Lon .< 0]
    maxlat = maximum(Lat)
    j1 = findfirst(lat .< maxlat)
    latsparse = lat[j1:end]
    colatsparse = colat[j1:end]
    nlatsparse = length(latsparse)

    U = zeros(T, nlon, nlatsparse, nt)
    B = zeros(T, nlon, nlatsparse, nt)
    SSH = zeros(T, nlon, nlatsparse, nt)
    H = zeros(T, nlon, nlatsparse, nt)
    R = zeros(T, 350, 350)
    for i in 1:nlon, j in 1:nlatsparse
        R .= sqrt.((lon[i] .- Lon).^2 + (latsparse[j] .- Lat).^2)
        IJ = argsmallest(R, 4)
        sumR = sum([R[IJ[l]] for l in 1:4])
        for l in 1:4
            w = R[IJ[l]] / sumR
            if (R[IJ[1]] < 0.5) || (j > nlatsparse - 80)
                U[i, j, :] += w * u[IJ[l], :]
                B[i, j, :] += w * b[IJ[l], :]
                SSH[i, j, :] += w * ssh[IJ[l], :]
                H[i, j, :] += w * Hice[IJ[l], :]
            end
        end
    end

    ds = NCDataset("$DIR/GaussLegendre-$FILE.nc", "c")
    defDim(ds, "lon", length(lon))
    defDim(ds, "colat", nlatsparse)
    defDim(ds, "t", nt)
    nclon = defVar(ds, "lon", T, ("lon",))
    nccolat = defVar(ds, "colat", T, ("colat",))
    nct = defVar(ds, "t", T, ("t",))
    nclon[:] = lon
    nccolat[:] = colatsparse
    nct[:] = t
    append3D2nc!(ds, T, U, "u")
    append3D2nc!(ds, T, B, "b")
    append3D2nc!(ds, T, SSH, "ssh")
    append3D2nc!(ds, T, H, "h")
    close(ds)
end

main()


ks = [64, 74, 82, 90, 98]   # -20, -15, -13, -11, -9
ks = [82, 98]
save2txt("$DIR/GaussLegendre-$FILE.nc", ks)