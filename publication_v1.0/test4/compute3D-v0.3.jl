push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")

#=
init()

N = 150
maxdepth = 500e3
nlayers = 3
use_cuda = true
interactive_sl = true

u = Omega.arraykernel(copy(fip.out.u[1]))
dudt = Omega.arraykernel(copy(fip.out.u[1]))
t = 0.0
update_diagnostics!(dudt, u, fip, t)
@btime update_diagnostics!($dudt, $u, $fip, $t)
@profview dudt_isostasy!(dudt, u, fip, t)
@code_warntype dudt_isostasy!(dudt, u, fip, t)
@btime dudt_isostasy!($dudt, $u, $fip, $t)
# On GPU:
# 428.753 μs (961 allocations: 56.81 KiB)
# On CPU:
# 135.883 μs (47 allocations: 641.05 KiB)
=#

function main(N, maxdepth, interactive_sl; nlayers = 3, use_cuda = false)
    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
    Lon, Lat = Omega.Lon, Omega.Lat
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
    #=
    if viscinterp == "log"
        lv_3D = 10 .^ cat([logeta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)
    elseif viscinterp == "exp"
        lv_3D = 10 .^ cat([logeta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)
    end
    =#
    eta_lowerbound = 1e16
    lv_3D[lv_3D .< eta_lowerbound] .= eta_lowerbound
    p = LayeredEarth(Omega, layer_viscosities = lv_3D, layer_boundaries = lb)

    (lon, lat, t), Hice, Hitp = load_ice6gd()
    Hice_vec = [Hitp.(Array(Omega.Lon), Array(Omega.Lat), tt) for tt in t]
    tsec = years2seconds.(t .* 1e3)

    fip = FastIsoProblem(Omega, c, p, tsec, interactive_sl, tsec,
        Hice_vec, verbose = true, b_0 = bathymetry(Omega))

    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    path = "../data/test4/ICE6G/3D-interactivesl=$interactive_sl-maxdepth=$maxdepth-"*
        "N=$(Omega.Nx)"
    @save "$path.jld2" t fip Hitp Hice_vec
    savefip("$path.nc", fip)
end

init()
for isl in [false]
    main(150, 300e3, isl, use_cuda = true)
end