using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

function main(N, maxdepth; nlayers = 3, use_cuda = false, interactive_sl = false)
    Omega = ComputationDomain(3300e3, 3300e3, N, N, use_cuda = use_cuda)
    Lon, Lat = Array(Omega.Lon), Array(Omega.Lat)
    c = PhysicalConstants()

    (_, _), Tpan, Titp = load_lithothickness_pan2022()
    Tlitho = Titp.(Lon, Lat) .* 1e3
    mindepth = maximum(Tlitho) + 1e3
    lb_vec = range(mindepth, stop = maxdepth, length = nlayers)
    lb = cat(Tlitho, [fill(lbval, Omega.Nx, Omega.Ny) for lbval in lb_vec]..., dims=3)
    rlb = c.r_equator .- lb
    nlb = size(rlb, 3)

    (_, _, _), _, logeta_itp = load_logvisc_pan2022()
    lv_3D = 10 .^ cat([logeta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)
    eta_lowerbound = 1e16
    lv_3D[lv_3D .< eta_lowerbound] .= eta_lowerbound
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv_3D)

    (lon, lat, t), Hice, Hitp = load_ice6gd()
    Hice_vec = [Hitp.(Array(Omega.Lon), Array(Omega.Lat), tt) for tt in t]
    tsec = years2seconds.(t .* 1e3)

    (x, y), b, bitp = load_dataset("BedMachine3")
    (x, y), geoid, geoiditp = load_dataset("BedMachine3", var = "geoid")
    fip = FastIsoProblem(Omega, c, p, tsec, interactive_sl, tsec, Hice_vec, verbose = true,
        b_0 = Omega.arraykernel(bitp.(Array(Omega.X), Array(Omega.Y))),
        seasurfaceheight_0 = Omega.arraykernel(geoiditp.(Array(Omega.X), Array(Omega.Y))),
        H_ice_0 = Hice_vec[1],
    )

    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    @save "../data/test4/ICE6G/3D-interactivesl=$interactive_sl-maxdepth=$maxdepth"*
        "-nlayers=$nlayers-N=$(Omega.Nx)-premparams.jld2" t fip Hitp Hice_vec
end

init()
[main(64, 300e3, interactive_sl = isl) for isl in [false, true]]