using CairoMakie
using DelimitedFiles
include("../helpers_plot.jl")

curdir = @__DIR__
comp = readdlm("$curdir/../../data/compressibility/decay.comp.out", ',', Float64, '\n')
incomp = readdlm("$curdir/../../data/compressibility/decay.incomp.out", ',', Float64, '\n')

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1], xlabel=L"Spherical harmonic degree $\,$", ylabel=L"Decay time (kyr) $\,$")
lines!(ax, view(comp, :, 1), view(comp, :, 2), label = L"compressible $\,$")
lines!(ax, view(incomp, :, 1), view(incomp, :, 2), label = L"incompressible $\,$")
axislegend(ax)
ax.xticks = (10:10:50, latexify(10:10:50))
ax.yticks = (4:8, latexify(4:8))
save("$curdir/../plots/compressibility.pdf", fig)