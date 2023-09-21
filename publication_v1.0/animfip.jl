function anim(path::String)

    @load "$path" fip
    extr = [maximum(abs.(extrema(u))) for u in (fip.out.u + fip.out.ue)]
    crange = (-maximum(extr), maximum(extr))

    kobs = Observable(1)
    u = @lift(fip.out.u[$kobs] + fip.out.ue[$kobs])
    fig = Figure(resolution = (1200, 900), fontsize = 30)
    ax = Axis3(fig[1, 1], title = @lift("t = $(round(seconds2years(fip.out.t[$kobs]))) yr") )
    hidedecorations!(ax)
    sf = surface!(ax, fip.Omega.X, fip.Omega.Y, u, colormap = :PuOr, colorrange = crange)
    wireframe!(ax, fip.Omega.X, fip.Omega.Y, u, linewidth = 0.05)
    Colorbar(fig[1, 2], sf, label = "Total displacement (m)", height = Relative(0.5))
    zlims!(ax, crange)
    record(fig, "anims/test.gif", eachindex(fip.out.t), framerate = 24) do k
        kobs[] = k
    end
end

filenames = ["../data/test1/Nx=64_Ny=64_cpu_interactive_sl=false-dense.jld2"]