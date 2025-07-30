using FastIsostasy
using JLD2, NCDatasets, Interpolations, DelimitedFiles
include("../helpers_computation.jl")
include("topography.jl")

function main(N, isl; use_cuda = false, mask_bsl = true)

    # Basics
    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
    Lon, Lat = Omega.Lon, Omega.Lat
    c = PhysicalConstants()
    opts = SolverOptions(interactive_sealevel = isl, verbose = true,
        internal_bsl_update = false, deformation_model = :lv_elra)

    # Load lithospheric thickness
    (_, _), _, Titp = load_lithothickness_pan2022()
    Tlitho = Titp.(Lon, Lat) .* 1e3
    mindepth = maximum(Tlitho) + 1e3
    lb = cat(Tlitho, fill(mindepth, Omega.nx, Omega.ny), dims=3)

    # Define upper-mantle ralxation time scale
    tau_wais, tau_eais = 2.0, 3.0
    _, _, mask_itp = load_dataset("Antarctic3RegionMask")
    mask_eais = mask_itp.(Omega.X .* 1e-3, Omega.Y .* 1e-3) .== 5
    sharp_tau = fill(tau_wais, Omega.nx, Omega.ny)
    sharp_tau[mask_eais] .= tau_eais
    tau = gaussian_smooth(sharp_tau, Omega, 0.2)
    tau[tau .< 2.0] .= tau_wais
    tau = years2seconds.(5.0 * 10 .^ (tau))
    p = LayeredEarth(Omega, tau = tau, layer_boundaries = lb)

    # Load ice thickness and deduce (active load) mask from it.
    (_, _, t), _, Hitp = load_ice6gd()
    Hice_vec = [Hitp.(Array(Omega.Lon), Array(Omega.Lat), tt) for tt in t]
    if isl
        k_lgm = argmax([mean(Hice_vec[k]) for k in eachindex(Hice_vec)])
        sharp_lgm_mask = Float64.(Hice_vec[k_lgm] .> 1e-3)
        blurred_lgm = gaussian_smooth(sharp_lgm_mask, Omega, 0.05)
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
    if mask_bsl
        southpole_msl_vec = [mean(sl_itp.(Omega.Lon, Omega.Lat, t) .*
            not.(blurred_lgm_mask)) for t in tlaty]
    else
        southpole_msl_vec = [mean(sl_itp.(Lon, Lat, t)) for t in tlaty]
    end
    bsl_itp = linear_interpolation(tlaty, southpole_msl_vec, extrapolation_bc = Flat())

    tsec = years2seconds.(t .* 1e3)
    fip = FastIsoProblem(Omega, c, p, tsec, tsec, Hice_vec, opts = opts, b_0 = bathy_0,
        bsl_itp = bsl_itp, maskactive = Omega.arraykernel(blurred_lgm_mask))

    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    dir = @__DIR__
    path = "$dir/../../data/test4/ICE6G/lvelra-interactivesl=$isl-maskbsl=$mask_bsl-"*
        "N=$(Omega.nx)"
    @save "$path.jld2" t fip Hitp Hice_vec
    savefip("$path.nc", fip)
end

init()
main(256, true, use_cuda = true)

# N = 350
# use_cuda = false
# Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
# (x, y), mask, mask_itp = load_dataset("Antarctic3RegionMask")
# mask_eais = mask_itp.(Omega.X .* 1e-3, Omega.Y .* 1e-3) .== 5
# sharp_tau = fill(1e2, Omega.nx, Omega.ny)
# sharp_tau[mask_eais] .= 3e3
# tau = gaussian_smooth(sharp_tau, Omega, 0.1)
# heatmap(tau)