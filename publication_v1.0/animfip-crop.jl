using FastIsostasy, JLD2, CairoMakie
include("helpers_plot.jl")

function simple_anim(path::String, target::String)

    @load "$path" fip
    extr = [maximum(abs.(extrema(u))) for u in (fip.out.u + fip.out.ue)]
    crange = (-maximum(extr), maximum(extr))

    kobs = Observable(1)
    u = @lift(fip.out.u[$kobs] + fip.out.ue[$kobs])

    fig = Figure(size = (1200, 900), fontsize = 30)
    with_theme(fig, theme_dark())
    ax = Axis3(fig[1, 1], title = @lift("t = $(round(seconds2years(fip.out.t[$kobs]))) yr") )
    hidedecorations!(ax)
    sf = surface!(ax, fip.Omega.X, fip.Omega.Y, u, colormap = :PuOr, colorrange = crange)
    wireframe!(ax, fip.Omega.X, fip.Omega.Y, u, linewidth = 0.05)
    Colorbar(fig[1, 2], sf, label = "Total displacement (m)", height = Relative(0.5))
    zlims!(ax, crange)
    record(fig, "anims/$target.gif", eachindex(fip.out.t), framerate = 24) do k
        kobs[] = k
    end
end

transient_extrema(u_vec) = maximum([maximum(abs.(extrema(u))) for u in u_vec])

function anim(path::String, target::String)

    @load "$path" fip
    t = fip.out.t
    umax = transient_extrema(fip.out.u + fip.out.ue)
    ulim = (-umax, umax)

    # grmask = [Union{Float64, missing}.(mask) for mask in fip.out.maskgrounded]
    # flmask = [Float64.(mask) for mask in fip.out.maskgrounded]
    Omega = fip.Omega
    grmask = [zeros(Union{Float64, Missing}, Omega.Nx, Omega.Ny) for _ in fip.out.t]
    flmask = [zeros(Union{Float64, Missing}, Omega.Nx, Omega.Ny) for _ in fip.out.t]

    for k in eachindex(grmask)
        mask = fip.out.maskgrounded[k]
        grmask[k][mask .> 0.5] .= 1.0
        grmask[k][mask .< 0.5] .= missing

        flmask[k][mask .< 0.5] .= 1.0
        flmask[k][mask .> 0.5] .= missing
    end

    kobs = Observable(1)
    u = @lift(fip.out.u[$kobs] + fip.out.ue[$kobs])
    ttl = @lift("Time t = $(round(seconds2years(fip.out.t[$kobs]))) yr")

    xyticks = (-3e6:1e6:3e6, latexify(-3:1:3))
    zticks = [(-2e3:1e3:2e3, latexify(-2:1:2)), (-6e2:2e2:6e2, latexify(-6:2:6))]
    fig = Figure(size = (950, 900), fontsize = 30)
    axs = [Axis3(fig[1, j], title = ttl, xticks = xyticks, yticks = xyticks,
        zticks = zticks[2]) for j in 1:1]
    colgap!(fig.layout, 5)

    sf2 = surface!(axs[1], Omega.X, Omega.Y, u, colormap = :PuOr, colorrange = ulim)
    Colorbar(fig[2, 1], sf2, label = "Bedrock displacement (m)", vertical = false,
        width = Relative(0.5))
    zlims!(axs[1], ulim)

    record(fig, "anims/$target.gif", eachindex(fip.out.t)[1:1:end], framerate = 18) do k
        kobs[] = k
    end
end

# Example 2
N = 350
filename = "../data/test4/ICE6G/3D-interactivesl=true-maskbsl=true-N=$N.jld2"
target = "test4/light-isl-ice6g-N=$N-crop"
anim() = anim(filename, target)
anim()
