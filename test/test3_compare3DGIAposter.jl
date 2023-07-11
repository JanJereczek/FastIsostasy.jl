push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles
include("helpers_plot.jl")

global include_elastic = true

function load_results(dir::String, idx)
    files = readdir(dir)
    nr = size( readdlm(joinpath(dir, files[1]), ','), 1 )
    u = zeros(nr, length(files))
    for i in eachindex(files)
        file = files[i]
        println( file, typeof( readdlm(joinpath(dir, file), ',')[:, 1] ) )
        u[:, i] = readdlm(joinpath(dir, file), ',')[:, 2]
    end

    u_plot = u[idx, :]

    if include_elastic
        u_3DGIA = u_plot
    else
        u_3DGIA = u_plot .- u_plot[:, 1]
    end

    return u_3DGIA
end

phi = -180:0.1:180
R = 6.371e6
r = R .* deg2rad.(phi)
idx = -3e6 .< r .< 3e6
r_plot = r[idx]

u_3DGIA = [load_results("data/Latychev/rt_E0L1V1_comma", idx),
    load_results("data/Latychev/rt_E0L2V1_comma", idx)]

n = 7
N = 2^n
kernel = "cpu"
suffix = "$(kernel)_Nx$(N)_Ny$(N)_dense"

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
x = Omega.X[slicey, slicex]

xlabels = [
    L"Position along great circle (m) $\,$",
    L"Position along great circle (m) $\,$",
]
ylabels = [
    L"$u_\mathrm{3DGIA} - u_\mathrm{fastiso}$ (m)",
    "",
]
yticklabelsvisible = [true, false]
idx = vcat(1:6, 7:2:15)
labels = [ L"t = %$t yr $\,$" for t in vcat( 0:1000:5000, 10000:5000:50000) ]

fig = Figure(resolution = (1600, 700), fontsize = 30)
axs = [Axis(
    fig[1, j],
    xlabel = xlabels[j],
    ylabel = ylabels[j],
    yticklabelsvisible = yticklabelsvisible[j],
) for j in eachindex(u_3DGIA)]
cmap = cgrad(:jet, 11, categorical = true)

for j in eachindex(u_3DGIA)
    for (i, k) in zip(eachindex(u_fastiso[j])[idx], eachindex(idx))
        itp = linear_interpolation(r_plot, u_3DGIA[j][:, i], extrapolation_bc = Flat())
        lines!(axs[j], x, itp.(x) - u_fastiso[j][i][slicey, slicex],
            color = cmap[k], label = labels[i])
    end
end
Legend(fig[:,3], axs[1])

utks = -15:5:25
ytks = ( utks, [L"%$u $\,$" for u in utks] )
rtks = -3e6:1e6:3e6
xtks = ( rtks, [L"%$r $\,$" for r in rtks] )

axs[1].title = L"Thin lithosphere $\,$"
axs[2].title = L"Thick lithosphere $\,$"
axs[2].yticksvisible = false
axs[1].xticks = xtks
axs[2].xticks = xtks
axs[1].yticks = ytks

ylims!(axs[1], (-15, 25))
ylims!(axs[2], (-15, 25))

fig
figfile = "plots/test3/posterfastiso3Dgia_elastic=$(include_elastic)_N=$(N)"
save("$figfile.png", fig)
save("$figfile.pdf", fig)