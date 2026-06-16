# FastIsostasyMakieExt

"""
    plot_transect(sim::Simulation, vars;
        ii = sim.domain.mx:sim.domain.nx,
        jj = sim.domain.my,
        analytic_cylinder_solution = false)

Plot the results of a FastIsostasy simulation along a transect. The transect is defined by the indices `ii` and `jj`, which specify the range of grid points to plot. The `vars` argument specifies which variables to plot (e.g., `:u`, `:h`, etc.). If `analytic_cylinder_solution` is set to `true`, the function will also plot the analytical solution for a cylindrical load.
"""
function plot_transect end
export plot_transect

"""
    plot_load(domain, H_ice)

Plot the ice load as a function of position. The `domain` argument specifies the spatial domain of the simulation, and `H_ice` is a matrix representing the ice thickness at each grid point.
"""
function plot_load end
export plot_load

"""
    plot_earth(sim::Simulation)
    plot_earth(domain, solidearth)

Plot the solid-Earth parameters used in the simulation.
"""
function plot_earth end
export plot_earth

"""
    plot_out_at_time(sim, vars, t)

Plot the output of a FastIsostasy simulation at a specific time `t`. The `vars` argument specifies which variables to plot.
"""
function plot_out_at_time end
export plot_out_at_time

"""
    plot_out_over_time(sim, var, t_vec, copts)

Plot the output of a FastIsostasy simulation for a specific variable `var` over a range of times specified in `t_vec`. The `copts` argument can be used to customize the plot (e.g., colors, line styles, etc.).
"""
function plot_out_over_time end
export plot_out_over_time

"""
    plot_computation_time(sim)

Plot the simulation year as a function of computation time. This can be useful for assessing the performance of the simulation.
"""
function plot_computation_time end
export plot_computation_time