module FastIsostasyMakieExt

using FastIsostasy, Makie
using DocStringExtensions

"""
$(TYPEDSIGNATURES)
"""
function FastIsostasy.plot_transect(sim::Simulation, vars; analytic_cylinder_solution = false)
    domain, c, p, t_out = sim.domain, sim.c, sim.p, sim.nout.t

    res_x, res_y = 400, 300
    set_theme!(theme_latexfonts())
    l1 = 3
    ii, jj = domain.mx:domain.nx, domain.my
    x, y = domain.X[ii, jj], domain.Y[ii, jj]
    cmap = cgrad(:jet, length(t_out), categorical = true)
    z = similar(x)
    k = 2 # Int(length(x) ÷ 1.2)

    fig = Figure(size = (res_x*length(vars), res_y))
    axs = [Axis(fig[1, j]) for j in eachindex(vars)]
    for j in eachindex(vars)
        axs[j].xlabel = L"$x \: (10^3 \: \mathrm{km})$"
        axs[j].title = "$(string((vars[j])))"

        for i in eachindex(t_out)
            t = t_out[i]

            if vars[j] == :u && analytic_cylinder_solution
                analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R)
                u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
                lines!(ax3, x, u_analytic, linestyle = :dash, color = cmap[i],
                    linewidth = l1)
            end

            z .= view(sim.nout.vals[vars[j]][i], ii, jj)
            lines!(axs[j], x, z, color = cmap[i], linewidth = l1, label = "$(t) yr")
            # text!(axs[j], x[k], z[k], text = L"$ %$t $ yr", color = cmap[i])
        end
        Legend(fig[1, length(vars)+1], axs[j])
        # axislegend(axs[j], position = :rc)
    end

    
    return fig
end

end # module