module FastIsostasyMakieExt

using FastIsostasy, Makie
using DocStringExtensions

"""
$(TYPEDSIGNATURES)
"""
function FastIsostasy.plot_transect(sim::Simulation, vars;
    ii = sim.domain.mx:sim.domain.nx,
    jj = sim.domain.my,
    analytic_cylinder_solution = false)

    domain, c, solidearth, t_out = sim.domain, sim.c, sim.solidearth, sim.nout.t

    res_x, res_y = 400, 300
    set_theme!(theme_latexfonts())
    l1 = 3
    x, y = domain.X[ii, jj], domain.Y[ii, jj]
    cmap = cgrad(:jet, length(t_out), categorical = true)
    z = similar(x)

    fig = Figure(size = (res_x*length(vars), res_y))
    axs = [Axis(fig[1, j]) for j in eachindex(vars)]
    for j in eachindex(vars)
        axs[j].xlabel = "x (km)"
        axs[j].title = "$(string((vars[j])))"

        for i in eachindex(t_out)
            t = t_out[i]

            if vars[j] == :u && analytic_cylinder_solution
                R0, H0 = 1f6, 1f3
                analytic_solution_r(r) = analytic_solution(
                    r,
                    t .* sim.c.seconds_per_year,
                    sim.c,
                    sim.solidearth,
                    H0,
                    R0,
                )
                u_analytic = analytic_solution_r.( sqrt.( x .^ 2 + y .^ 2 ) )
                lines!(axs[j], x ./ 1f3, u_analytic, linestyle = :dash, color = cmap[i],
                    linewidth = l1)
            end

            z .= view(sim.nout.vals[vars[j]][i], ii, jj)
            lines!(axs[j], x ./ 1f3, z, color = cmap[i], linewidth = l1,
                label = "$(Int(round(t))) yr")
        end
        Legend(fig[1, length(vars)+1], axs[j])
    end

    
    return fig
end

function FastIsostasy.plot_load(domain, H_ice)
    set_theme!(theme_latexfonts())
    fig = Figure(size = (500, 400))
    ax = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, domain.x ./ 1f3, domain.y ./ 1f3, H_ice, colormap = :ice)
    Colorbar(fig[1, 2], hm, label = "Ice thickness (m)", height = Relative(0.6))
    return fig
end

function FastIsostasy.plot_earth(sim)
    return FastIsostasy.plot_earth(sim.domain, sim.solidearth)
end

function FastIsostasy.plot_earth(domain, solidearth)
    ncols = 3
    set_theme!(theme_latexfonts())
    fig = Figure(size = (1000, 400))
    axs = [Axis(fig[1, j], aspect = DataAspect()) for j in 1:ncols]
    [hidedecorations!(axs[j]) for j in 1:ncols]

    x = domain.x ./ 1f3
    y = domain.y ./ 1f3
    T_litho = solidearth.litho_thickness ./ 1f3
    log10eta = log10.(solidearth.effective_viscosity)
    kappa = solidearth.pseudodiff_scaling
    
    hm1 = heatmap!(axs[1], x, y, T_litho)
    if maximum(T_litho) == minimum(T_litho)
        axs[1].title = "Constant lithosphere thickness $(T_litho[1])"
    else
        Colorbar(fig[0, 1], hm1, label = "Lithosphere thickness (km)", vertical = false, 
            width = Relative(0.6))
        contour!(axs[1], x, y, solidearth.maskactive, color = :red,
            linewidth = 3, levels = [0.5])
    end

    hm2 = heatmap!(axs[2], x, y, log10eta)
    if maximum(log10eta) == minimum(log10eta)
        axs[2].title = "Constant effective viscosity $(log10eta[1])"
    else
        Colorbar(fig[0, 2], hm2, label = "Effective viscosity (log10(Pa s))", vertical = false, 
            width = Relative(0.6))
        contour!(axs[2], x, y, solidearth.maskactive, color = :red,
            linewidth = 3, levels = [0.5])
    end

    hm3 = heatmap!(axs[3], x, y, kappa)
    if maximum(kappa) == minimum(kappa)
        axs[3].title = "Constant pseudodiff scaling $(kappa[1])"
    else
        Colorbar(fig[0, 3], hm3, label = "Pseudodiff scaling (1)", vertical = false, width = Relative(0.6))
    end

    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    return fig
end

function FastIsostasy.plot_out_at_time(sim, vars, t)
    res = 300
    x, y = xy_km(sim.domain)
    k = argmin(abs.(sim.nout.t .- t))
    Z = similar(sim.domain.X)
    set_theme!(theme_latexfonts())
    fig = Figure(size = (res * length(vars), res))

    for j in eachindex(vars)
        ax = Axis(fig[1, j], aspect = DataAspect())
        hidedecorations!(ax)
        if vars[j] == :u_tot
            Z .= sim.nout.vals[:u][k] .+ sim.nout.vals[:ue][k]
        else
            Z .= sim.nout.vals[vars[j]][k]
        end
        heatmap!(ax, x, y, Z; colormap = :jet)
    end

    Colorbar(fig[0, length(vars)+1], vertical = false,
        label = "$(vars) at t = $(Int(round(t))) yr", height = Relative(0.6))
    
    return fig
end

function FastIsostasy.plot_out_over_time(sim, var, t_vec, copts)
    res = 300
    x, y = xy_km(sim.domain)
    nt = length(t_vec)
    Z = similar(sim.domain.X)
    set_theme!(theme_latexfonts())
    fig = Figure(size = (res * nt, res))

    for j in eachindex(t_vec)
        ax = Axis(fig[1, j], aspect = DataAspect())
        hidedecorations!(ax)
        t = t_vec[j]
        k = argmin(abs.(sim.nout.t .- t))
        if var == :u_tot
            Z .= sim.nout.vals[:u][k] .+ sim.nout.vals[:ue][k]
        else
            Z .= sim.nout.vals[var][k]
        end
        heatmap!(ax, x, y, Z; copts...)
        ax.title = "$(var) at t = $(Int(round(t))) yr"
    end

    Colorbar(fig[1, nt+1], label = "$(var)", height = Relative(0.6); copts...)
    
    return fig
end

xy_km(domain::RegionalDomain) = (domain.x ./ 1f3, domain.y ./ 1f3)

# lines(u_x[domain.mx:end, domain.my], color = :black, linewidth = 1)
# lines(u_y[domain.mx:end, domain.my], color = :black, linewidth = 1)
# s = 4
# arrows(domain.x[1:s:end], domain.y[1:s:end], u_x[1:s:end, 1:s:end],
#     u_y[1:s:end, 1:s:end], arrowsize = 3, lengthscale = 1e4,
#     arrowcolor = :gray10, linecolor = :gray10)
# heatmap(u_h)

end # module