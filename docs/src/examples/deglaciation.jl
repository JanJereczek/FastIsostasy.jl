#=
# Antarctic deglaciation

We now want to provide an example that presents:
- a heterogeneous lithosphere thickness
- a heterogeneous upper-mantle viscosity
- various viscous channels
- a more elaborate load that evolves over time
- changes in the sea-level

For this we run a deglaciation of Antarctica with lithospheric thickness and upper-mantle viscosity from [^Wiens2021] and the ice thickness history from [^Briggs2014]. Since the load is known and the isostatic response does not influence it (one-way coupling), we can provide snapshots of the ice thickness and their associated time to [`FastIsoProblem`](@ref). Under the hood, an interpolator is created and called within the time integration. 
=#

using CairoMakie, FastIsostasy
## Code is coming soon!

#=
[^Wiens2021]:
    Douglas Wiens et al. (2021): [The seismic structure of the Antarctic upper mantle](https://doi.org/10.1144/M56-2020-18)

[^Briggs2014]:
    Robert Briggs et al. (2014): [A data constrained large ensemble analysis of Antarctic evolution since the Eemian](https://doi.org/10.1016/j.quascirev.2014.09.003)

=#