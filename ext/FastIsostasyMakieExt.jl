module FastIsostasyMakieExt

using FastIsostasy, Makie
using DocStringExtensions

"""
$(TYPEDSIGNATURES)
"""
function FastIsostasy.plot_transect(fip::FastIsoProblem, vars; analytic_cylinder_solution = false)
    Omega, c, p, t_out = fip.Omega, fip.c, fip.p, fip.nout.t[2:end]

    set_theme!(theme_latexfonts())
    l1 = 3
    ii, jj = 1:Omega.mx, Omega.my
    x, y = Omega.X[ii, jj], Omega.Y[ii, jj]
    cmap = cgrad(:jet, length(t_out), categorical = true)
    z = similar(x)
    k = Int(length(x) ÷ 1.2)

    fig = Figure()
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

            z .= view(fip.nout.vals[vars[j]][i], ii, jj)
            lines!(axs[j], x, z, color = cmap[i], linewidth = l1)
            text!(axs[j], x[k], z[k], text = L"$ %$t $ yr", color = cmap[i])
        end
    end

    
    return fig
end

end # module