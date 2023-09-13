push!(LOAD_PATH, "../")
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles

datasets = ["ICE6G", "ICE7G"]
prefixes = ["I6_C.VM5a_1deg.", "I7G_NA.VM7_1deg."]

chooseidx = 1
dataset = datasets[chooseidx]
prefix = prefixes[chooseidx]

dir = "../data/$dataset"
files = readdir(dir)
t = Float64[]
for file in files
    tstring = chop(file, head = length(prefix), tail = length(".nc"))
    push!(t, parse(Float64, tstring))
end
idx = sortperm(-t)
t = -t[idx]
files = files[idx]
nt = length(t)

lon = -179:180
lat = -89:90
nlon, nlat = length(lon), length(lat)
topodiff = zeros(nlon, nlat, nt)
topo = zeros(nlon, nlat, nt)
ice_thickness = zeros(nlon, nlat, nt)

for k in eachindex(t)
    NCDataset(joinpath(dir, files[k])) do ds
        topodiff[:, :, k] = ds["Topo_Diff"][:, :]
        topo[:, :, k] = ds["Topo"][:, :]
        ice_thickness[:, :, k] = ds["stgit"][:, :]
    end
end
Hsum = [sum(ice_thickness[:, :, k]) for k in eachindex(t)]

Hlim = (1e-8, 4e3)
fig = Figure(resolution = (1600, 1000), fontsize = 24)
ax1 = Axis(fig[1:3, 1], aspect = DataAspect())
ax2 = Axis(fig[4, 1])
hidedecorations!(ax1)
xlims!(ax2, extrema(t))
ylims!(ax2, extrema(Hsum) .+ (-2e6, 2e6))

kobs = Observable(1)
Hobs = @lift( ice_thickness[:, :, $kobs] )
nobs = @lift( length(t[1:$kobs]) )
tobs = @lift( t[1:$nobs] )
Vproxy = @lift( Hsum[1:$nobs] )
heatmap!(ax1, Hobs, colormap = :ice, colorrange = Hlim, lowclip = :transparent)
points = Observable(Point2f[(t[1], Hsum[1])])
lines!(ax2, points)

record(fig, "$dataset.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
    new_point = Point2f(t[k], Hsum[k])
    points[] = push!(points[], new_point)
end

Hitp = linear_interpolation((lon, lat, t), ice_thickness, extrapolation_bc = Flat())
using FastIsostasy

Omega = ComputationDomain(3000e3, 7)
c = PhysicalConstants()
lbmantle = c.r_equator .- [5.721, 5.179] .* 1e6
lb = vcat(96e3, lbmantle)
lv = [0.5, 1.58, 3.16] .* 1e21
# lb = [96e3]
# lv = [5e20]
p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)

Hice = [copy(Omega.null) for _ in t]
for k in eachindex(t)
    for IJ in CartesianIndices(Omega.Lat)
        i, j = Tuple(IJ)
        Hice[k][i, j] = Hitp(Omega.Lon[i, j], Omega.Lat[i, j], t[k])
    end
end
deltaH = [Hice[k] - Hice[1] for k in eachindex(t)]

tsec = years2seconds.(t .* 1e3)
fip = FastIsoProblem(Omega, c, p, tsec, false, tsec, deltaH)
solve!(fip)
println("Computation took $(fip.out.computation_time) s")

kobs = Observable(1)
fig = Figure(resolution = (1600, 1000), fontsize = 30)
axs = [Axis(fig[1, j], title = @lift("t = $(t[$kobs]) ka"),
    aspect = DataAspect()) for j in 1:2]
[hidedecorations!(ax) for ax in axs]
hm1 = heatmap!(axs[1], @lift(Hice[$kobs]), colorrange = Hlim, lowclip = :transparent, colormap = :ice)
hm2 = heatmap!(axs[2], @lift(fip.out.u[$kobs]), colormap = :vik, colorrange = (-400, 400))
Colorbar(fig[2, 1], hm1, vertical = false, flipaxis = false, width = Relative(0.8))
Colorbar(fig[2, 2], hm2, vertical = false, flipaxis = false, width = Relative(0.8))
record(fig, "$dataset-displacement.mp4", eachindex(t), framerate = 10) do k
    kobs[] = k
end


#=
latydir = "../data/Latychev/ICE6G/r1d_ICE6G"
latyfiles = readdir(latydir)
tlaty = [-120, -26, -21, -16, -11, -6, -2, -1, -0.125, 0, 0.125] .* 1e3

latyfiles = latyfiles[2:end-1]
tlaty = tlaty[2:end-1]
=#




















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