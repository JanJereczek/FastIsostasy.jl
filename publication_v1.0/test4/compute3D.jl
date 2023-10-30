push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2, NCDatasets, CairoMakie, Interpolations, DelimitedFiles
using CairoMakie
include("../helpers.jl")

function lon360tolon180(lon, X)
    permidx = lon .> 180
    lon180 = vcat(lon[permidx] .- 360, lon[not.(permidx)])
    Hice180 = cat(X[permidx, :, :], X[not.(permidx), :, :], dims=1)
    return lon180, Hice180
end

function load_laty_3Dvisc()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Latychev/ICE6G/dense/stp_model_MH"
    data = vec(readdlm(file, skipstart = 2))
    r = data[1:64]
    eta_ratio_vec = data[65:end]

    lon = collect(range(0, stop = 360, length = 721))
    lat = collect(range(90, stop = -90, length = 361))
    nr, nlat, nlon = length(r), length(lat), length(lon)
    eta_ratio = reshape(eta_ratio_vec, (nlon, nlat, nr))
    heatmap(eta_ratio[:, :, end])
    reverse!(eta_ratio, dims = 2)
    reverse!(lat)
    heatmap(eta_ratio[:, :, end])
    eta1D = [3.16, 3.16, 1.58, 1.58, 0.5, 0.5] .* 1e21
    r_eta1D = [3480000.0, 5173800.9, 5173800.95, 5701000.0, 5701000.05, 6371000.0]
    eta1D_itp = linear_interpolation(r_eta1D, eta1D)
    eta1D_3D = zeros((nlon, nlat, nr))
    for i in axes(eta1D_3D, 1), j in axes(eta1D_3D, 2)
        eta1D_3D[i, j, :] .= eta1D_itp.(r)
    end
    eta = eta1D_3D .* (10 .^ eta_ratio)
    lon180, eta180 = lon360tolon180(lon, eta)
    heatmap(eta180[:, :, end-5])
    eta_itp = linear_interpolation((lon180, lat, r), eta180, extrapolation_bc = Flat())
    return eta_itp
end

function main(N, maxdepth; nlayers = 3, use_cuda = true, interactive_sl = false)
    Omega = ComputationDomain(3500e3, 3500e3, N, N, use_cuda = use_cuda)
    Lon, Lat = Array(Omega.Lon), Array(Omega.Lat)
    c = PhysicalConstants(rho_uppermantle = 3.3e3, rho_litho = 2.7e3)

    Titp = load_litho_thickness_laty()
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
    eta_itp = load_laty_3Dvisc()
    lv_3D = cat([eta_itp.(Lon, Lat, rlb[:, :, k]) for k in 1:nlb]..., dims=3)
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

    forcing = "ICE6G_D"
    if forcing == "ICE6G_C"
        Hice, t, lat, lon = load_ice6g()
        t .*= -1
        lon180, Hice180 = lon360tolon180(lon, Hice)
        Hitp = linear_interpolation((lon180, lat, t), Hice180, extrapolation_bc = Flat())
    elseif forcing == "ICE6G_D"
        t, lon, lat, Hice, Hitp = load_ice6gd()
    end
    Hice_vec, deltaH = vec_dHice(Omega, Lon, Lat, t, Hitp)

    tsec = years2seconds.(t .* 1e3)
    fip = FastIsoProblem(Omega, c, p, tsec, interactive_sl, tsec, deltaH, verbose = true)

    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")

    @save "../data/test4/ICE6G/3D-interactivesl=$interactive_sl-maxdepth=$maxdepth"*
        "-nlayers=$nlayers-$forcing-N=$(Omega.Nx).jld2" t fip Hitp Hice_vec deltaH
end

function load_litho_thickness_laty()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Latychev/ICE6G/dense/LAB.llz"
    data, head = readdlm(file, header = true)
    Lon_vec, Lat_vec, T_vec = data[:, 1], data[:, 2], data[:, 3]
    lon, lat = unique(Lon_vec), unique(Lat_vec)
    nlon, nlat = length(lon), length(lat)
    Lon = reshape(Lon_vec, nlon, nlat)
    Lat = reshape(Lat_vec, nlon, nlat)
    Tlitho = reshape(T_vec, nlon, nlat)
    reverse!(Tlitho, dims=2)
    reverse!(Lat, dims=2)
    reverse!(lat)
    heatmap(Tlitho)
    lon180, Tlitho180 = lon360tolon180(lon, Tlitho)
    itp = linear_interpolation((lon180, lat), Tlitho180[:, :, 1], extrapolation_bc = Flat())
    return itp
end

init()
main(350, 300e3, use_cuda = true, interactive_sl = false)

# for maxdepth in 300:100:600
#     main(64, maxdepth * 1e3, use_cuda = false)
# end


# eta_ratio_safe = zeros((nlon, nlat, nr))
# for idx in CartesianIndices(eta_ratio_safe)
#     i, j, k = Tuple(idx)
#     eta_ratio_safe[i, j, k] = eta_ratio_vec[i + (j-1)*nlon + (k-1)*nlat*nlon]
# end

# ulaty = reverse(ulaty, dims = 2)