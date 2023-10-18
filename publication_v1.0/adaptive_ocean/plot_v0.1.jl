using FastIsostasy, CairoMakie
include("helpers.jl")

osc = OceanSurfaceChange()
z = -150:0.1:70
srf = osc.A_itp.(z)

Vice = 45e6 * (1e3)^3
nV = 10000
dV = Vice / nV
Vvec = -dV:-dV:-Vice
zvec = zeros(nV)
osc = OceanSurfaceChange()
for i in 1:nV
    osc(-dV)
    zvec[i] = osc.zk
end
hA0vec = Vvec ./ osc.A_pd

lw = 3
fig = Figure(resolution = (1200, 500), fontsize = 25)
ax = Axis(fig[1:6,1],
    xlabel = L"Barystatic SL contribution $SLC$ (m)",
    ylabel = L"Ocean surface $A(SLC)$ ($\times 10^{14} \: \mathrm{m}^2$)",
    xticks = latexticks(-150:50:100),
    yticks = ((3.4:0.1:3.8) .* 1e14, latexify(3.4:0.1:3.8)),
)
xlims!(ax, (-150, 70))
hlines!(ax, [osc.A_pd], linewidth = lw)
lines!(ax, z, srf, linewidth = lw, color = :orange)

ax2col = :gray60
ax = Axis(fig[1:6, 2],
    yaxisposition = :right,
    xlabel = L"$\Delta V$ of ice above floatation ($\times 10^{16} \: \mathrm{m}^3$)",
    ylabel = L"Barystatic SL (m) $\,$",
    xticks = ( (-4:1:0) .* 1e16, latexify(-4:1:0)),
    yticks = latexticks(-120:20:0))
lines!(ax, Vvec, hA0vec, linewidth = lw, label = L"Fixed boundaries $\,$")
lines!(ax, Vvec, zvec, color = :orange, linewidth = lw, label = L"Trapezoidal approximation $\,$")
hlines!(ax, [1e20], color = ax2col, linewidth = lw, label = L"Difference $\,$")
xlims!(ax, (-4.5e16, 0))
ylims!(ax, (-130, 0))

Legend(fig[7,:], ax, nbanks = 3)

ax = Axis(fig[1:6, 2],
    ytickcolor = ax2col,
    yticklabelcolor = ax2col,
    yticks = latexticks(0:6),
    ygridvisible = false,
    ylabelcolor = ax2col,
    ylabel = L"Difference (m) $\,$",
)
xlims!(ax, (-130, 0))
hidexdecorations!(ax)
lines!(ax, hA0vec, hA0vec - zvec, color = ax2col, linewidth = lw)
fig

save("plots/adaptive_ocean/lines_ocean_nonlin.pdf", fig)