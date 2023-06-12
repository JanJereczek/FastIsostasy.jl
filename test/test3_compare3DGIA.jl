push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles
include("helpers_plot.jl")

include_elastic = true

dir = "data/Latychev_R20"
files = readdir(dir)
nr = size( readdlm(joinpath(dir, files[1]), '\n'), 1 )
u = zeros(nr, length(files))
for i in eachindex(files)
    file = files[i]
    u[:, i] = readdlm(joinpath(dir, file), '\n')
end
phi = -180:0.1:180
R = 6.371e6
r = R .* deg2rad.(phi)
idx = -3e6 .< r .< 3e6
u_plot = u[idx, :]
r_plot = r[idx]

if include_elastic
    u_3DGIA = [ u_plot[:, 1:5], u_plot[:, 6:10] ]
else
    u_3DGIA = [ u_plot[:, 1:5] .- u_plot[:, 1], u_plot[:, 6:10] .- u_plot[:, 6] ]
end

n = 6
N = 2^n
kernel = "cpu"
suffix = "$(kernel)_N$(N)_dense"
function get_denseoutput_fastiso(suffix)
    sol_lo_D = load("data/test3/gaussian_lo_D_$suffix.jld2")
    sol_hi_D = load("data/test3/gaussian_hi_D_$suffix.jld2")
    sols = [sol_lo_D, sol_hi_D]
    results = [sol["results"] for sol in sols]
    if include_elastic
        u_plot = [res.viscous + res.elastic for res in results]
    else
        u_plot = [res.viscous for res in results]
    end
    return u_plot, sol_lo_D["Omega"]

end
u_fastiso, Omega = get_denseoutput_fastiso(suffix)
n1, n2 = size(u_fastiso[1][1])
slicey, slicex = Int(n1/2), 1:n2

xlabels = [
    L"Position along great circle (m) $\,$",
    L"Position along great circle (m) $\,$",
]
ylabels = [
    L"Vertical viscous displacement (m) $\,$",
    "",
]
yticklabelsvisible = [true, false]
labels = [
    L"t = 0 yr $\,$",
    L"t = 500 yr $\,$",
    L"t = 1000 yr $\,$",
    L"t = 1500 yr $\,$",
    L"t = 2000 yr $\,$",
]
titles = [
    L"Thin lithosphere $\,$",
    L"Thick lithosphere $\,$",
]
fig = Figure()
axs = [Axis(
    fig[1,j],
    xlabel = xlabels[j],
    ylabel = ylabels[j],
    yticklabelsvisible = yticklabelsvisible[j],
    title = titles[j],
) for j in eachindex(u_3DGIA)]
for j in eachindex(u_3DGIA)
    for i in 1:5
        lines!(axs[j], r_plot, u_3DGIA[j][:, i], color = Cycled(i), label = labels[i])
        lines!(axs[j], Omega.X[slicey, slicex], u_fastiso[j][i][slicey, slicex],
            linestyle = :dash, color = Cycled(i))
    end
end
axislegend(axs[1])
# Legend(fig[2,:], , vertical = false)
fig
save("plots/test3/fastiso3Dgia_elastic=$(include_elastic).png", fig)
save("plots/test3/fastiso3Dgia_elastic=$(include_elastic).pdf", fig)