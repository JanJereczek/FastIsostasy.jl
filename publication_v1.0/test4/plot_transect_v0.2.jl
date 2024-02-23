using FastIsostasy, CairoMakie
using NCDatasets, LinearAlgebra

function crossection(x, ii, jj)
    crossx = zeros(eltype(x), length(ii), size(x, 3))
    for i in eachindex(ii)
        crossx[i, :] = x[ii[i], jj[i], :]
    end
    return crossx
end

function main(transect)
    # Load fields
    file = "../data/test4/ICE6G/3D-interactivesl=true-bsl=external-N=350.nc"
    ds = NCDataset(file)
    t = copy(ds["t"][:])
    x = copy(ds["x"][:])
    y = copy(ds["y"][:])
    b = copy(ds["b"][:, :, :])
    ssh = copy(ds["seasurfaceheight"][:, :, :])
    geoid = copy(ds["geoid"][:, :, :])
    maskgrounded = copy(ds["maskgrounded"][:, :, :])
    Hice = copy(ds["Hice"][:, :, :])
    Hwater = copy(ds["Hwater"][:, :, :])
    close(ds)
    H_soutpole = [mean(Hice[173:4:177, 173:4:177, k]) for k in axes(Hice, 3)]
    for i in 174:176, j in 174:176
        Hice[i, j, :] .= H_soutpole
        maskgrounded[i, j, :] .= 1f0
    end

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
            xsupport = x[40:end-25]
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

    sparsity = 30
    sparsity_indices = 1:sparsity:length(ii)-sparsity
    ii_sparse = ii[sparsity_indices]
    jj_sparse = jj[sparsity_indices]
    rr = sqrt.((x[ii] .- x[ii[1]]) .^ 2 + (y[jj] .- y[jj[1]]) .^ 2)
    rlolim, rhilim = extrema(rr)

    # Preprocess fields
    maskocean = Hwater .> 0
    zsbottom = fill(1e30, size(Hice))
    zstop = fill(-1e30, size(Hice))
    zsbottom[maskocean] .= (ssh[maskocean] - Hice[maskocean]) .* (931 / 1028)
    zstop[maskocean] .= zsbottom[maskocean] + Hice[maskocean]
    maskshelves = zsbottom .< Inf32
    sshmask = fill(-Inf32, size(ssh)...)
    sshmask[maskgrounded .== 0] = ssh[maskgrounded .== 0]
    Vaprox = [sum(Hice[:, :, k]) for k in eachindex(t)]

    bcross = crossection(b, ii, jj)
    Hcross = crossection(Hice, ii, jj)
    maskgroundedcross = crossection(maskgrounded, ii, jj)
    sshcross = crossection(ssh, ii, jj)
    geoidcross = crossection(geoid, ii, jj)
    sshmaskcross = crossection(sshmask, ii, jj)
    zsbottomcross = crossection(zsbottom, ii, jj)
    zstopcross = crossection(zstop, ii, jj)

    # Fields for animation
    kobs = Observable(1)
    H = @lift(Hice[20:end-20, 20:end-20, $kobs])
    zz = @lift(bcross[:, $kobs] + Hcross[:, $kobs] .* maskgroundedcross[:, $kobs])
    bb = @lift(bcross[:, $kobs])
    ss = @lift(sshcross[:, $kobs])
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

    # Figure
    fig = Figure(size = (800, 600), fonstize = 20)

    dist = rlolim:0.5e6:rhilim
    distticks = (dist, string.(round.(dist ./ 1e6, digits = 1)))
    dist_sparse = rlolim:1e6:rhilim
    distticks_sparse = (dist_sparse, string.(round.(dist_sparse ./ 1e6, digits = 1)))

    # SSH perturbation axis
    sshax = Axis(fig[2:3, 1], xgridvisible = false, ygridvisible = false,
        xaxisposition = :top, xticks = distticks_sparse,
        xlabel = L"Distance from cross-section beginning ($10^3$ km)",
        ylabel = L"Sea-surface perturbation (m) $\,$")
    xlims!(sshax, (rlolim, rhilim))
    ylims!(sshax, (-10, 10))
    lines!(sshax, rr, gg, color = :dodgerblue4)

    # Map axis
    mapax = Axis(fig[2:3, 2], aspect = DataAspect())
    hidedecorations!(mapax)
    hm = heatmap!(mapax, x, y, H, colorrange = (1e-8, 4e3), colormap = cgrad(:blues, rev = true),
        lowclip = :transparent, highclip = :lightblue1)
    Colorbar(fig[1, 2], hm, vertical = false, width = Relative(0.8), label = L"Ice thickness (m) $\,$")

    lines!(mapax, x[ii], y[jj], color = :red, linewidth = 3)
    greymap = cgrad([:black, :white], eachindex(sparsity_indices) ./ length(sparsity_indices))
    # scatter!(mapax, x[ii_sparse], y[jj_sparse], color = greymap[eachindex(sparsity_indices)],
    #     markersize = 20)

    # Cycle axis
    cycleax = Axis(fig[2:3, 3], xaxisposition = :top, yaxisposition = :right,
        xlabel = L"Time before present (kyr) $\,$",
        ylabel = L"Integrated ice thickness (m) $\,$")
    lines!(cycleax, t ./ 1e3, Vaprox, color = :gray10)
    scatter!(cycleax, tt, VV, color = :gray10)

    # Main axis
    ax = Axis(fig[4:6, :], xgridvisible = false, ygridvisible = false,
        xlabel = L"Distance from cross-section beginning ($10^3$ km) $\,$", xticks = distticks,
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
    axislegend(ax, position = :lt)

    # scatter!(mapax, rr[eachindex(sparsity_indices)], zeros(length(sparsity_indices)),
    #     color = greymap[eachindex(sparsity_indices)], markersize = 20)

    # fig[0, 1:3] = Label(fig, ttl)
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    rowgap!(fig.layout, 1, -30)
    rowgap!(fig.layout, 2, 0)

    fig

    record(fig, "anims/transect_shelves_$transect.mp4", eachindex(t), framerate = 10) do k
        kobs[] = k
    end
end

transects = ["greenwich", "rossronne", "thwaitesamery"]
for transect in ["rossronne"]
    main(transect)
end