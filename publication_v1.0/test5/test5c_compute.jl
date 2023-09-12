push!(LOAD_PATH, "../")
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles

file = "../data/ICE6Gzip/IceT.I6F_C.131QB_VM5a_1deg.nc"
ds = NCDataset(file)
Hice = copy(ds["stgit"][:, :])
t = copy(ds["Time"][:, :])
lat = copy(ds["Lat"][:, :])
lon = copy(ds["Lon"][:, :]) .- 180
close(ds)
t .*= -1
nlon, nlat, nt = size(Hice)
Hsum = [sum(Hice[:, :, k]) for k in eachindex(t)]

Hlim = (1e-8, 4e3)
fig = Figure(resolution = (1600, 1000), fontsize = 24)
ax1 = Axis(fig[1:3, 1], aspect = DataAspect())
ax2 = Axis(fig[4, 1])
hidedecorations!(ax1)
xlims!(ax2, extrema(t))
ylims!(ax2, extrema(Hsum) .+ (-2e6, 2e6))

kobs = Observable(1)
Hobs = @lift( Hice[:, :, $kobs] )
nobs = @lift( length(t[1:$kobs]) )
tobs = @lift( t[1:$nobs] )
Vproxy = @lift( Hsum[1:$nobs] )
heatmap!(ax1, Hobs, colormap = :ice, colorrange = Hlim, lowclip = :transparent)
points = Observable(Point2f[(t[1], Hsum[1])])
lines!(ax2, points)

record(fig, "ICE6G-cylce.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
    new_point = Point2f(t[k], Hsum[k])
    points[] = push!(points[], new_point)
end

Hitp = linear_interpolation((lon, lat, t), Hice, extrapolation_bc = Flat())
using FastIsostasy

Omega = ComputationDomain(3500e3, 3500e3, 200, 200)
c = PhysicalConstants()
lbmantle = c.r_equator .- [5.721, 5.179] .* 1e6
lb = vcat(96e3, lbmantle)
lv = [0.5, 1.58, 3.16] .* 1e21
# lb = [96e3]
# lv = [5e20]
p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)

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
fip = FastIsoProblem(Omega, c, p, tsec, false, tsec, deltaH)
solve!(fip)
println("Computation took $(fip.out.computation_time) s")

kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
axs = [Axis(fig[1, j], title = @lift("t = $(t[$kobs]) ka"),
    aspect = DataAspect()) for j in 1:2]
[hidedecorations!(ax) for ax in axs]
hm1 = heatmap!(axs[1], @lift(Hice_vec[$kobs]), colorrange = Hlim, lowclip = :transparent, colormap = :ice)
hm2 = heatmap!(axs[2], @lift(fip.out.u[$kobs]), colormap = :vik, colorrange = (-800, 800))
Colorbar(fig[2, 1], hm1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], hm2, vertical = false, flipaxis = false, width = Relative(0.8))
record(fig, "ICE6G-cycle-displacement.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
end

dHlims = (-2600, 2600)
ulims = (-500, 500)
kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
vars = [deltaH, fip.out.u .+ fip.out.ue]
axs = [Axis3(fig[1, j], title = @lift("t = $(t[$kobs]) ka,"*
 "  range = $(round.(extrema(vars[j][$kobs])))")) for j in 1:2]
# [hidedecorations!(ax) for ax in axs]
sf1 = surface!(axs[1], @lift(deltaH[$kobs]), colorrange = dHlims, colormap = :vik)
sf2 = surface!(axs[2], @lift(fip.out.u[$kobs] + fip.out.ue[$kobs]), colormap = :PuOr,
    colorrange = ulims)
zlims!(axs[1], dHlims)
zlims!(axs[2], ulims)
Colorbar(fig[2, 1], sf1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], sf2, vertical = false, flipaxis = false, width = Relative(0.8))
record(fig, "ICE6G-cycle-surface.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
end


latydir = "../data/Latychev/ICE6G/r1d_ICE6G"
latyfiles = readdir(latydir)
tlaty = [-120, -26, -21, -16, -11, -6, -2, -1, -0.125, 0, 0.125] .* 1e3

latyfiles = latyfiles[1:end-1]
tlaty = tlaty[1:end-1]
for latyfile in latyfiles
    displ = readdlm(joinpath(latydir, latyfile))
    println(extrema(displ))
end



















#=
# landarea_fraction = zeros(nlon, nlat, nt)
# icearea_fraction = zeros(nlon, nlat, nt)
# orog = zeros(nlon, nlat, nt)

# landarea_fraction[:, :, k] = ds["sftlf"][:, :]
# icearea_fraction[:, :, k] = ds["sftgif"][:, :]
# orog[:, :, k] = ds["orog"][:, :]

using FastIsostasy
include("../test/helpers/compute.jl")
include("../test/helpers/loadmaps.jl")
include("../test/helpers/viscmaps.jl")

function main(n::Int, active_gs::Bool; use_cuda::Bool = false,solver = "SimpleEuler")

    T = Float64
    W = T(4000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n)   # domain parameters
    c = PhysicalConstants()

    lb = [88e3, 180e3, 280e3, 400e3]
    halfspace_logviscosity = fill(21.0, Omega.Nx, Omega.Ny)
    Eta, Eta_mean, z = load_wiens2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)

    kernel = use_cuda ? "gpu" : "cpu"
    println("Computing on $kernel and $(Omega.Nx) x $(Omega.Ny) grid...")

    t_out, deltaH, H = interpolated_glac1d_snapshots(Omega)
    dH = [deltaH[:, :, k] for k in axes(deltaH, 3)]
    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, t_out, dH,
        interactive_sealevel = active_gs,
        alg=solver,
        dt = years2seconds(0.1),
    )
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    if use_cuda
        Omega, p = reinit_structs_cpu(Omega, p)
    end

    case = active_gs ? "geostate" : "isostate"
    jldsave(
        "../data/test5/$(case)_Nx$(Omega.Nx)_Ny$(Omega.Ny).jld2",
        Omega = Omega, c = c, p = p,
        results = results,
        t_fastiso = t_fastiso,
        H = dH,
    )
end

cases = [false, true]
for active_gs in cases[1:1]
    main(6, active_gs, use_cuda=false, solver="SimpleEuler")
end
=#