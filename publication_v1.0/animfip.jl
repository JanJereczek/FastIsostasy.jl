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
    dH = [fip.out.Hice[k] - fip.out.Hice[1] for k in eachindex(t)]
    umax = transient_extrema(fip.out.u + fip.out.ue)
    Hmax = transient_extrema(dH)
    ulim = (-umax, umax)
    Hlim = (-Hmax, Hmax)
    sshlim = transient_extrema(fip.out.seasurfaceheight)

    # grmask = [Union{Float64, missing}.(mask) for mask in fip.out.maskgrounded]
    # flmask = [Float64.(mask) for mask in fip.out.maskgrounded]
    Omega = fip.Omega
    zeros = zeros(Omega.nx, Omega.ny)
    grmask = [zeros(Union{Float64, Missing}, Omega.nx, Omega.ny) for _ in fip.out.t]
    flmask = [zeros(Union{Float64, Missing}, Omega.nx, Omega.ny) for _ in fip.out.t]

    for k in eachindex(grmask)
        mask = fip.out.maskgrounded[k]
        grmask[k][mask .> 0.5] .= 1.0
        grmask[k][mask .< 0.5] .= missing

        flmask[k][mask .< 0.5] .= 1.0
        flmask[k][mask .> 0.5] .= missing
    end

    kobs = Observable(1)
    u = @lift(fip.out.u[$kobs] + fip.out.ue[$kobs])
    # b = @lift(fip.out.b[$kobs])
    H = @lift(dH[$kobs])
    # z = @lift( (fip.out.b[$kobs] + fip.out.Hice[$kobs]) .* grmask[$kobs])
    ssh = @lift(fip.out.seasurfaceheight[$kobs] .* flmask[$kobs])
    ttl = @lift("Time t = $(round(seconds2years(fip.out.t[$kobs]))) yr")

    xyticks = (-3e6:1e6:3e6, latexify(-3:1:3))
    zticks = [(-2e3:1e3:2e3, latexify(-2:1:2)), (-6e2:2e2:6e2, latexify(-6:2:6))]
    fig = Figure(size = (1900, 900), fontsize = 30)
    axs = [Axis3(fig[1, j], title = ttl, xticks = xyticks, yticks = xyticks,
        zticks = zticks[j]) for j in 1:2]
    colgap!(fig.layout, 5)

    sf1 = surface!(axs[1], Omega.X, Omega.Y, H, colormap = :balance, colorrange = Hlim)
    sf2 = surface!(axs[2], Omega.X, Omega.Y, u, colormap = :PuOr, colorrange = ulim)
    # wireframe!(axs[2], Omega.X, Omega.Y, u, linewidth = 0.05)
    Colorbar(fig[2, 1], sf1, label = "Ice thickness anomaly (m)", vertical = false,
        width = Relative(0.5))
    Colorbar(fig[2, 2], sf2, label = "Bedrock displacement (m)", vertical = false,
        width = Relative(0.5))
    zlims!(axs[1], Hlim)
    zlims!(axs[2], ulim)

    record(fig, "anims/$target.gif", eachindex(fip.out.t)[1:1:end], framerate = 18) do k
        kobs[] = k
    end
end


# Example 1
#=
filenames = ["../data/test1/nx=64_Ny=64_cpu_interactive_sl=false-dense.jld2",
    "../data/test4/ICE6G/heterogeneous-interactivesl=true-N=128.jld2"]
targets = ["nx=64_Ny=64_cpu_interactive_sl=false-dense",
    "test4/heterogeneous-interactivesl=true-N=128"]
simple_anim(filenames[2], targets[2])
=#

# Example 2
N = 350
filename = "../data/test4/ICE6G/3D-interactivesl=true-maskbsl=true-N=$N.jld2"
target = "test4/light-isl-ice6g-N=$N"
anim() = anim(filename, target)
anim()
# with_theme(anim, theme_dark())
# with_theme(anim, theme_dark())
