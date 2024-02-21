using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
include("../helpers.jl")
include("topography.jl")

function main(N, maxdepth, isl; nlayers = 3, use_cuda = false)

    # Basics
    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
    Lon, Lat = Omega.Lon, Omega.Lat
    c = PhysicalConstants()
    lbmantle = c.r_equator .- [5.721, 5.179] .* 1e6
    lb = vcat(96e3, lbmantle)
    lv = [0.5, 1.58, 3.16] .* 1e21
    p = LayeredEarth(Omega, layer_boundaries = lb[1:2], layer_viscosities = lv[1:2])
    opts = SolverOptions(interactive_sealevel = isl, verbose = true, internal_bsl_update = false)

    # Load ice thickness and deduce (active load) mask from it.
    (_, _, t), _, Hitp = load_ice6gd()
    Hice_vec = [Hitp.(Array(Omega.Lon), Array(Omega.Lat), tt) for tt in t]
    # South pole gives 0 because of projection. Correct by mean smoothing.
    H_soutpole = [mean(H[173:4:177, 173:4:177]) for H in Hice_vec]
    for k in eachindex(Hice_vec)
        Hice_vec[k][174:176, 174:176] .= H_soutpole[k]
    end

    if isl
        k_lgm = argmax([mean(Hice_vec[k]) for k in eachindex(Hice_vec)])
        sharp_lgm_mask = Float64.(Hice_vec[k_lgm] .> 1e-3)
        blurred_lgm = blur(sharp_lgm_mask, Omega, 0.05)
        blurred_lgm_mask = blurred_lgm .> 0.5 * maximum(blurred_lgm)
    else
        blurred_lgm_mask = Omega.X .< Inf
    end

    # Load topography
    _, _, topo_itp = load_latychev_topo()
    bathy_0 = Omega.arraykernel(topo_itp.(Omega.Lon, Omega.Lat))

    # Load barystatic sea level (not global one, since we are interested in SH)
    (lonlaty, latlaty, tlaty), sl, sl_itp = load_latychev2023_ICE6G(case = "3D", var = "SL")
    Lon, Lat = meshgrid(lonlaty, latlaty[1:40])
    southpole_msl_vec = [mean(sl_itp.(Lon, Lat, t)) for t in tlaty]
    bsl_itp = linear_interpolation(tlaty, southpole_msl_vec, extrapolation_bc = Flat())
    # gmslvec = [mean(sl[:, :, k]) for k in axes(sl, 3)]
    # _, _, bsl_itp = load_bsl()

    tsec = years2seconds.(t .* 1e3)
    fip = FastIsoProblem(Omega, c, p, tsec, tsec, Hice_vec, opts = opts, b_0 = bathy_0,
        bsl_itp = bsl_itp, maskactive = Omega.arraykernel(blurred_lgm_mask))

    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    path = "../data/test4/ICE6G/newconv-1D-interactivesl=$isl-bsl=external-"*
        "N=$(Omega.Nx)"
    @save "$path.jld2" t fip Hitp Hice_vec
    savefip("$path.nc", fip)
end

init()
for isl in [true]
    main(350, 300e3, isl, use_cuda = true)
end





#=
init()

n = 7
maxdepth = 500e3
nlayers = 3
use_cuda = true
isl = true
Omega = ComputationDomain(3500e3, n, use_cuda = use_cuda)
c = PhysicalConstants()
p = LayeredEarth(Omega)
opts = SolverOptions(interactive_sealevel = isl, verbose = true)
(lon, lat, t), Hice, Hitp = load_ice6gd()
Hice_vec = [Hitp.(Array(Omega.Lon), Array(Omega.Lat), tt) for tt in t]
tsec = years2seconds.(t .* 1e3)
mask = collect(Omega.R .< 1e6)
fip = FastIsoProblem(Omega, c, p, tsec, tsec, Hice_vec, opts = opts, maskactive = mask)

u = Omega.arraykernel(copy(fip.out.u[1]))
dudt = Omega.arraykernel(copy(fip.out.u[1]))
t = 0.0
update_diagnostics!(dudt, u, fip, t)
@btime update_diagnostics!($dudt, $u, $fip, $t)
@profview update_diagnostics!(dudt, u, fip, t)

@profview dudt_isostasy!(dudt, u, fip, t)
@code_warntype dudt_isostasy!(dudt, u, fip, t)
@btime dudt_isostasy!($dudt, $u, $fip, $t)

# On GPU:
# 428.753 μs (961 allocations: 56.81 KiB)

# On GPU, with apply_bc!
416.316 μs (782 allocations: 45.12 KiB)

# On CPU:
# 135.883 μs (47 allocations: 641.05 KiB)
=#
