push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

function main(N)
    # Hice, t, lat, lon = load_ice6g()
    # lon180, Hice180 = lon360tolon180(lon, Hice)
    # fig, ax, hm = heatmap(Hice180[:, :, 2], colormap = :ice, colorrange = (1e-8, 4e3),
    #     lowclip = :transparent)
    # Colorbar(fig[1, 2], hm)
    # hidedecorations!(ax)
    # fig

    # lon, Hice = lon180, Hice180
    # t .*= -1
    # nlon, nlat, nt = size(Hice)
    # Hsum = [sum(Hice[:, :, k]) for k in eachindex(t)]

    # Hlim = (1e-8, 4e3)
    # fig = Figure(resolution = (1600, 1000), fontsize = 24)
    # ax1 = Axis(fig[1:3, 1], aspect = DataAspect())
    # ax2 = Axis(fig[4, 1])
    # hidedecorations!(ax1)
    # hideydecorations!(ax2)
    # xlims!(ax2, extrema(t))
    # ylims!(ax2, extrema(Hsum) .+ (-2e6, 2e6))

    # kobs = Observable(1)
    # Hobs = @lift( Hice[:, :, $kobs] )
    # nobs = @lift( length(t[1:$kobs]) )
    # tobs = @lift( t[1:$nobs] )
    # Vproxy = @lift( Hsum[1:$nobs] )
    # hm = heatmap!(ax1, Hobs, colormap = :ice, colorrange = Hlim, lowclip = :transparent)
    # Colorbar(fig[1:3, 2], hm, height = Relative(0.6))
    # points = Observable(Point2f[(t[1], Hsum[1])])
    # lines!(ax2, points)

    # record(fig, "plots/test4/ICE6G-cylce.mp4", eachindex(t), framerate = 10) do k
    #     kobs[] = k
    #     new_point = Point2f(t[k], Hsum[k])
    #     points[] = push!(points[], new_point)
    # end

    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = true)
    c = PhysicalConstants()
    lbmantle = c.r_equator .- [5.721, 5.179] .* 1e6
    lb = vcat(96e3, lbmantle)
    lv = [0.5, 1.58, 3.16] .* 1e21
    p = LayeredEarth(Omega, layer_boundaries = lb[1:2], layer_viscosities = lv[1:2])

    Lon, Lat = Array(Omega.Lon), Array(Omega.Lat)
    t, lon, lat, Hice, Hitp = load_ice6gd()
    Hice_vec, deltaH = vec_dHice(Omega, Lon, Lat, t, Hitp)

    tsec = years2seconds.(t .* 1e3)
    interactive_sl = false
    fip = FastIsoProblem(Omega, c, p, tsec, interactive_sl, tsec, deltaH)
    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    @save "../data/test4/ICE6G/1D-interactivesl=$interactive_sl-N="*
        "$(Omega.Nx)-premparams.jld2" t fip Hitp Hice_vec deltaH
end

main(350)