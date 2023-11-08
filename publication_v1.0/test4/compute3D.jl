push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
using CairoMakie
include("../helpers.jl")

function main(N, maxdepth; nlayers = 3, use_cuda = true, interactive_sl = false)
    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
    Lon, Lat = Array(Omega.Lon), Array(Omega.Lat)
    c = PhysicalConstants()
    # c = PhysicalConstants(rho_uppermantle = 3.3e3, rho_litho = 2.7e3)

    (_, _), Tpan, Titp = load_lithothickness_pan2022()
    Tlitho = Titp.(Lon, Lat) .* 1e3
    fig, ax, hm = heatmap(Tlitho)
    Colorbar(fig[1, 2], hm)
    save("plots/test4/laty_lithothickness.pdf", fig)

    # lb_vec = [96, 250, 400, 600] .* 1e3
    # lb = cat([fill(lbval, Omega.Nx, Omega.Ny) for lbval in lb_vec]..., dims=3)
    mindepth = maximum(Tlitho) + 1e3
    lb_vec = range(mindepth, stop = maxdepth, length = nlayers)

    # lb_vec = [250, 400, 600] .* 1e3
    lb = cat(Tlitho, [fill(lbval, Omega.Nx, Omega.Ny) for lbval in lb_vec]..., dims=3)
    rlb = c.r_equator .- lb
    nlb = size(rlb, 3)
    # nlb = 3  c.r_equator - lb_vec
    (_, _, _), _, logeta_itp = load_logvisc_pan2022()
    lv_3D = 10 .^ cat([logeta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)
    eta_lowerbound = 1e16
    lv_3D[lv_3D .< eta_lowerbound] .= eta_lowerbound

    rsltn = 1000
    fig = Figure(resolution = ( (nlb+1) * rsltn, rsltn), fontsize = 30)
    crange = (18, 23)
    cmap = cgrad(:jet, rev=true)
    for j in 1:nlb
        ax = Axis(fig[1, j], aspect = DataAspect())
        heatmap!(ax, log10.(lv_3D[:, :, j]), colormap = cmap, colorrange = crange)
        hidedecorations!(ax)
    end
    Colorbar(fig[1, nlb+2], colorrange = crange, colormap = cmap)
    
    # lb = cat(lb, fill(700e3, Omega.Nx, Omega.Ny), dims=3)
    # lv_3D = cat(lv_3D, fill(1e21, Omega.Nx, Omega.Ny), dims=3)
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv_3D)
    axeff = Axis(fig[1, nlb+1], aspect = DataAspect())
    heatmap!(axeff, log10.(Array(p.effective_viscosity)), colormap = cmap,
        colorrange = crange)
    hidedecorations!(axeff)
    save("plots/test4/viscmap_laty-maxdepth=$maxdepth.pdf", fig)

    (lon, lat, t) Hice, Hitp = load_ice6gd()
    Hice_vec, deltaH = vec_dHice(Omega, Lon, Lat, t, Hitp)

    tsec = years2seconds.(t .* 1e3)
    fip = FastIsoProblem(Omega, c, p, tsec, interactive_sl, tsec, deltaH, verbose = true)

    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    @save "../data/test4/ICE6G/3D-interactivesl=$interactive_sl-maxdepth=$maxdepth"*
        "-nlayers=$nlayers-N=$(Omega.Nx)-premparams.jld2" t fip Hitp Hice_vec deltaH
end


init()
main(350, 300e3, use_cuda = true, interactive_sl = false)