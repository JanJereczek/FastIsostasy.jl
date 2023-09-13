push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles

not(x::Bool) = !x

function lon360tolon180(lon, X)
    permidx = lon .> 180
    lon180 = vcat(lon[permidx] .- 360, lon[not.(permidx)])
    Hice180 = cat(X[permidx, :, :], X[not.(permidx), :, :], dims=1)
    return lon180, Hice180
end

function compute5C()
    file = "../data/ICE6Gzip/IceT.I6F_C.131QB_VM5a_1deg.nc"
    ds = NCDataset(file)
    Hice = copy(ds["stgit"][:, :])
    t = copy(ds["Time"][:, :])
    lat = copy(ds["Lat"][:, :])
    lon = copy(ds["Lon"][:, :])
    close(ds)

    lon180, Hice180 = lon360tolon180(lon, Hice)
    fig, ax, hm = heatmap(Hice180[:, :, 2], colormap = :ice, colorrange = (1e-8, 4e3),
        lowclip = :transparent)
    Colorbar(fig[1, 2], hm)
    hidedecorations!(ax)
    fig

    lon, Hice = lon180, Hice180
    t .*= -1
    nlon, nlat, nt = size(Hice)
    Hsum = [sum(Hice[:, :, k]) for k in eachindex(t)]

    Hlim = (1e-8, 4e3)
    fig = Figure(resolution = (1600, 1000), fontsize = 24)
    ax1 = Axis(fig[1:3, 1], aspect = DataAspect())
    ax2 = Axis(fig[4, 1])
    hidedecorations!(ax1)
    hideydecorations!(ax2)
    xlims!(ax2, extrema(t))
    ylims!(ax2, extrema(Hsum) .+ (-2e6, 2e6))

    kobs = Observable(1)
    Hobs = @lift( Hice[:, :, $kobs] )
    nobs = @lift( length(t[1:$kobs]) )
    tobs = @lift( t[1:$nobs] )
    Vproxy = @lift( Hsum[1:$nobs] )
    hm = heatmap!(ax1, Hobs, colormap = :ice, colorrange = Hlim, lowclip = :transparent)
    Colorbar(fig[1:3, 2], hm, height = Relative(0.6))
    points = Observable(Point2f[(t[1], Hsum[1])])
    lines!(ax2, points)

    record(fig, "plots/test5/ICE6G-cylce.mp4", eachindex(t), framerate = 10) do k
        kobs[] = k
        new_point = Point2f(t[k], Hsum[k])
        points[] = push!(points[], new_point)
    end

    Hitp = linear_interpolation((lon, lat, t), Hice, extrapolation_bc = Flat())

    Omega = ComputationDomain(3500e3, 7)
    c = PhysicalConstants()
    lbmantle = c.r_equator .- [5.721, 5.179] .* 1e6
    lb = vcat(96e3, lbmantle)
    lv = [0.5, 1.58, 3.16] .* 1e21
    p = LayeredEarth(Omega, layer_boundaries = lb[1:2], layer_viscosities = lv[1:2])

    Hice_vec = [copy(Omega.null) for _ in t]
    for k in eachindex(t)
        for IJ in CartesianIndices(Omega.Lat)
            i, j = Tuple(IJ)
            # println(Omega.Lon[i, j], "  ", Omega.Lat[i, j], "  ", t[k])
            Hice_vec[k][i, j] = Hitp(Omega.Lon[i, j], Omega.Lat[i, j], t[k])
        end
    end
    deltaH = [Hice_vec[k] - Hice_vec[1] for k in eachindex(t)]

    tsec = years2seconds.(t .* 1e3)
    interactive_sl = true
    fip = FastIsoProblem(Omega, c, p, tsec, interactive_sl, tsec, deltaH)
    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    @save "../data/test5/ICE6G/homogeneous-interactivesl=$interactive_sl-N="*
        "$(Omega.Nx).jld2" t fip Hitp Hice_vec deltaH
end