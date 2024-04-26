using FastIsostasy, CairoMakie, Colors
using NCDatasets, LinearAlgebra, JLD2
include("../helpers_plot.jl")

function crossection(x, ii, jj)
    crossx = zeros(eltype(x), length(ii), size(x, 3))
    for i in eachindex(ii)
        crossx[i, :] = x[ii[i], jj[i], :]
    end
    return crossx
end

function load_geom(N)
    Omega = ComputationDomain(3500e3, 3500e3, N, N)
    return Omega.Dx, Omega.Dy, Omega.K
end

function main(transect)
    # Load fields
    N = 350
    file = "../data/test4/ICE6G/3D-interactivesl=true-maskbsl=true-N=$N-2.nc"
    Dx, Dy, K = load_geom(N)
    ds = NCDataset(file)
    t = copy(ds["t"][:])
    x = copy(ds["x"][:])
    y = copy(ds["y"][:])
    bsl_vec = copy(ds["barystatic sea level"][:])
    b = copy(ds["b"][:, :, :])
    ssh = copy(ds["seasurfaceheight"][:, :, :])
    geoid = copy(ds["geoid"][:, :, :])
    maskgrounded = copy(ds["maskgrounded"][:, :, :])
    Hice = copy(ds["Hice"][:, :, :])
    Hwater = copy(ds["Hwater"][:, :, :])
    close(ds)

    # Fill south pole
    H_soutpole = [mean(Hice[173:4:177, 173:4:177, k]) for k in axes(Hice, 3)]
    for i in 174:176, j in 174:176
        Hice[i, j, :] .= H_soutpole
        maskgrounded[i, j, :] .= 1f0
    end

    # Get transect
    X, Y = meshgrid(x, y)
    if transect in ["rossronne", "thwaitesamery"]
        if transect == "rossronne"
            zlolim = -2000
            zhilim = 4500

            # Define transect with 2 points
            xp = [-1.2e6, 0]
            yp = [1.5e6, -1.9e6]
            XX = hcat(xp, ones(length(xp)))'
            M = inv(XX * XX' ) * XX
            m, p = M * yp
            xtransect = collect(xp[1]:5e3:xp[2])
            ytransect = m * xtransect .+ p
        elseif transect == "thwaitesamery"
            zlolim = -2000
            zhilim = 4500

            alpha = π / 8
            M = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
            # xsupport = x[40:end-25]
            xsupport = x[50:end-25]
            ysupport = zeros(length(xsupport))
            Mtransect = M * vcat(xsupport', ysupport')
            xtransect, ytransect = view(Mtransect, 1, :), view(Mtransect, 2, :)
        end
        ii = zeros(Int, length(xtransect))
        jj = zeros(Int, length(xtransect))
        cartesian_transect = CartesianIndex[]
        for i in eachindex(xtransect)
            cindex = argmin( (xtransect[i] .- X).^2 + (ytransect[i] .- Y).^2 )
            iijj = Tuple(cindex)
            push!(cartesian_transect, cindex)
            ii[i] = iijj[1]
            jj[i] = iijj[2]
        end
    elseif transect == "greenwich"
        zlolim = -1000
        zhilim = 3500

        ycrop_width = 40
        jj = ycrop_width:length(y)-ycrop_width
        ii = fill(argmin(abs.(x .+ 0.1e6)), length(jj))
    end

    # sparsity = 30
    # sparsity_indices = 1:sparsity:length(ii)-sparsity
    # ii_sparse = ii[sparsity_indices]
    # jj_sparse = jj[sparsity_indices]
    rr = sqrt.((x[ii] .- x[ii[1]]) .^ 2 + (y[jj] .- y[jj[1]]) .^ 2)
    rlolim, rhilim = extrema(rr)

    # Preprocess fields
    maskocean = Hwater .> 0
    zsbottom = fill(0f0, size(Hice))
    zstop = fill(0f0, size(Hice))
    zsbottom[maskocean] .= (ssh[maskocean] - Hice[maskocean]) .* (931 / 1028)
    zstop[maskocean] .= zsbottom[maskocean] + Hice[maskocean]

    # maskshelves = zsbottom .< Inf32
    sshmask = fill(-Inf32, size(ssh)...)
    sshmask[maskgrounded .== 0] = ssh[maskgrounded .== 0]

    volume(H) = sum(H .* Dx .* Dy) .* 1e-9 .* 1e-6         # result in millions km^3
    Vaprox = [volume(Hice[:, :, k]) for k in eachindex(t)]

    bcross = crossection(b, ii, jj)
    Hcross = crossection(Hice, ii, jj)
    maskgroundedcross = crossection(maskgrounded, ii, jj)
    geoidcross = crossection(geoid, ii, jj)
    sshmaskcross = crossection(sshmask, ii, jj)
    zsbottomcross = crossection(zsbottom, ii, jj)
    zstopcross = crossection(zstop, ii, jj)

    # Fields for animation
    kobs = Observable(1)
    H = @lift(Hice[20:end-20, 20:end-20, $kobs])
    zz = @lift(bcross[:, $kobs] + Hcross[:, $kobs] .* maskgroundedcross[:, $kobs])
    bb = @lift(bcross[:, $kobs])
    gg = @lift(geoidcross[:, $kobs])
    ssmask = @lift(sshmaskcross[:, $kobs])
    zzloclip = @lift(min.($zz, $bb))
    zzhiclip = @lift(max.($zz, $bb))
    bbloclip = @lift(min.(1.1*zlolim, $bb))
    bbhiclip = @lift(max.(1.1*zlolim, $bb))
    sshiclip = @lift(max.($ssmask, $bb))
    zshelfbottom = @lift(zsbottomcross[:, $kobs])
    zshelftop = @lift(zstopcross[:, $kobs])
    tt = @lift(t[$kobs] ./ 1e3)
    VV = @lift(Vaprox[$kobs])
    bsl = @lift(bsl_vec[$kobs])

    # Figure
    fig = Figure(size = (800, 600), fonstize = 20)

    defblue = RGBA{Float32}(0.0f0, 0.44705883f0, 0.69803923f0, 1.0f0)
    deforange = RGBA{Float32}(0.9019608f0,0.62352943f0,0.0f0,1.0f0)

    dist = rlolim:0.5e6:rhilim
    distticks = (dist, string.(round.(dist ./ 1e6, digits = 1)))
    dist_sparse = rlolim:1e6:rhilim
    distticks_sparse = (dist_sparse, string.(round.(dist_sparse ./ 1e6, digits = 1)))

    # Volume axis
    xvticks = collect(-100:50:0)
    vticks = collect(26:2:34)
    vcolor = defblue
    volax = Axis(fig[2:3, 1], xaxisposition = :top, yaxisposition = :left,
        xlabel = L"Time before present (kyr) $\,$",
        ylabel = L"AIS volume ($\mathrm{10^6 \, km^3}$)",
        ylabelcolor = vcolor, yticks = (vticks, latexify(vticks)),
        ytickcolor = vcolor, yticklabelcolor = vcolor,
        ygridvisible = false, xgridvisible = false,
        xticks = (xvticks, latexify(xvticks)))
    lines!(volax, t ./ 1e3, Vaprox, color = vcolor)
    scatter!(volax, tt, VV, color = vcolor)
    ylims!(volax, (26, 34))

    # BSL axis
    bslticks = collect(-85:25:0)
    bslcolor = deforange
    bslax = Axis(fig[2:3, 1], xaxisposition = :top, yaxisposition = :left,
        ylabel = L"Mean regional SL (m) $\,$", ylabelcolor = bslcolor,
        ytickcolor = bslcolor, yticklabelcolor = bslcolor,
        ygridvisible = false, xgridvisible = false,
        yticks = (bslticks, latexify(bslticks)), ylabelpadding = 20)
    hidexdecorations!(bslax)
    lines!(bslax, t ./ 1e3, bsl_vec, color = bslcolor)
    scatter!(bslax, tt, bsl, color = bslcolor)
    ylims!(bslax, (-98, 5))

    # Map axis
    mapax = Axis(fig[2:3, 2], aspect = DataAspect())
    hidedecorations!(mapax)
    hm = heatmap!(mapax, x, y, H, colorrange = (1e-8, 4e3), colormap = :ice,
        lowclip = :transparent, highclip = :white)
    Colorbar(fig[1, 2], hm, vertical = false, width = Relative(0.8),
        label = L"Ice thickness (km) $\,$",
        ticks = (vcat(1e-9, 1:4) .* 1e3, latexify(0:4)))

    lines!(mapax, x[ii], y[jj], color = :red, linewidth = 3)
    # greymap = cgrad([:black, :white], eachindex(sparsity_indices) ./ length(sparsity_indices))
    # scatter!(mapax, x[ii_sparse], y[jj_sparse], color = greymap[eachindex(sparsity_indices)],
    #     markersize = 20)

    # SSH perturbation axis
    xsshticks = collect(0:5)
    ysshticks = collect(-10:5:10)
    sshax = Axis(fig[2:3, 3],
        xgridvisible = false, ygridvisible = false,
        xaxisposition = :top, yaxisposition = :right,
        xlabel = L"Distance along transect ($10^3$ km)",
        ylabel = L"Sea-surface perturbation (m) $\,$",
        xticks = (xsshticks .* 1e6, latexify(xsshticks)),
        yticks = (ysshticks, latexify(ysshticks)))
    xlims!(sshax, (rlolim, rhilim))
    ylims!(sshax, (-10, 10))
    lines!(sshax, rr, gg, color = defblue)

    # Main axis
    xticks = collect(0.5:0.5:5)
    yticks = collect(-2000:1000:4500)
    ax = Axis(fig[4:6, :], xgridvisible = false, ygridvisible = false,
        xlabel = L"Distance along transect ($10^3$ km) $\,$",
        xticks = (xticks .* 1e6, latexify(xticks)),
        yticks = (yticks, latexify(yticks)),
        ylabel = L"Height (m) $\,$")
    xlims!(ax, (rlolim, rhilim))
    ylims!(ax, (zlolim, zhilim))

    lines!(ax, rr, ssmask, color = :dodgerblue4)
    band!(ax, rr, bb, sshiclip, color = :royalblue, label = L"Open ocean $\,$")
    band!(ax, rr, zzloclip, zzhiclip, color = :skyblue1, label = L"Grounded ice $\,$")
    band!(ax, rr, zshelfbottom, zshelftop, color = :lightskyblue1, label = L"Floating ice $\,$")
    lines!(ax, rr, zz, color = :deepskyblue1, linewidth = 3)

    lines!(ax, rr, bcross[:, 1], color = :saddlebrown, linewidth = 1, linestyle = :dash)
    lines!(ax, rr, bb, color = :saddlebrown, linewidth = 3)
    band!(ax, rr, bbloclip, bbhiclip, color = :tan1, label = L"Bedrock $\,$")
    axislegend(ax, position = :lt, nbanks = 2, framevisible = false)

    # scatter!(mapax, rr[eachindex(sparsity_indices)], zeros(length(sparsity_indices)),
    #     color = greymap[eachindex(sparsity_indices)], markersize = 20)

    # fig[0, 1:3] = Label(fig, ttl)
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    rowgap!(fig.layout, 1, -30)
    rowgap!(fig.layout, 2, 0)

    fig

    record(fig, "anims/transect_$transect-slow.mp4", eachindex(t), framerate = 5) do k
        kobs[] = k
    end
end

transects = ["greenwich", "rossronne", "thwaitesamery"]
for transect in transects[3:3]
    main(transect)
end