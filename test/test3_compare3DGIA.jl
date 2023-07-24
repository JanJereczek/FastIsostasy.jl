push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles
include("helpers_plot.jl")

n = 7
heterogeneous = "none"
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

function get_denseoutput_fastiso(fastiso_files)
    sols = [load("data/test3/$file") for file in fastiso_files]
    results = [sol["results"] for sol in sols]
    if include_elastic
        u_plot = [res.viscous + res.elastic for res in results]
    else
        u_plot = [res.viscous for res in results]
    end
    return u_plot, sols[1]["Omega"]

end

phi = -180:0.1:180
R = 6.371e6
r = R .* deg2rad.(phi)
idx = -1 .< r .< 3e6
r_plot = r[idx]

N = 2^n
kernel = "cpu"
suffix = "$(kernel)_Nx$(N)_Ny$(N)_dense"

if heterogeneous == "lithosphere"
    seakon_files = ["rt_E0L1V1_comma", "rt_E0L2V1_comma"]
    fastiso_files = ["gaussian_lo_D_$suffix.jld2", "gaussian_hi_D_$suffix.jld2"]
    elims = (-30, 30)
    title1 = L"Gaussian thinning lithosphere $\,$"
    title2 = L"Gaussian thickening lithosphere $\,$"
elseif heterogeneous == "upper-mantle"
    seakon_files = ["E0L3V2", "E0L3V3"]
    fastiso_files = ["gaussian_lo_η_$suffix.jld2", "gaussian_hi_η_$suffix.jld2"]
    elims = (-30, 30)
    title1 = L"Gaussian decrease of viscosity $\,$"
    title2 = L"Gaussian increase of viscosity $\,$"
elseif heterogeneous == "none"
    seakon_files = ["E0L4V4", "E0L4V4"]
    fastiso_files = ["ref_$suffix.jld2", "no_litho_$suffix.jld2"]
    elims = (-20, 45)
    title1 = L"Homogeneous PREM configuration $\,$"
    title2 = L"No-lithosphere configuration $\,$"
end

u_fastiso, Omega = get_denseoutput_fastiso(fastiso_files)
u_3DGIA = [load_results("data/Latychev/$file", idx) for file in seakon_files]

n1, n2 = size(u_fastiso[1][1])
slicey, slicex = n1÷2, n2÷2:n2
x = Omega.X[slicey, slicex]

xlabels = [
    L"Position along great circle (m) $\,$",
    L"Position along great circle (m) $\,$",
]

ylabels = [
    L"Vertical displacement (m) $\,$",
    L"$u_\mathrm{Seakon} - u_\mathrm{FastIso}$ (m)",
]

yticklabelsvisible = [true, false]
labels = [ L"t = %$t kyr $\,$" for t in vcat(0:1:5, 10:5:50) ]
cmap = cgrad(:jet, length(labels), categorical = true)
# cvec = [:cornflowerblue, :royalblue, :mediumpurple, :rebeccapurple, :gray10]
# cmap = cgrad(cvec, length(labels), categorical = true)

fig = Figure(resolution = (3200, 2000), fontsize = 60)
axs = [Axis(
    fig[i, j],
    xlabel = xlabels[j],
    ylabel = ylabels[i],
    yticklabelsvisible = yticklabelsvisible[j],
) for j in eachindex(u_3DGIA), i in 1:2]
for j in eachindex(u_3DGIA)
    for i in eachindex(u_fastiso[j])
        itp = linear_interpolation(r_plot, u_3DGIA[j][:, i], extrapolation_bc = Flat())
        lines!(axs[j], r_plot, u_3DGIA[j][:, i], color = cmap[i],
            label = labels[i], linewidth = 5)
        lines!(axs[j], x, u_fastiso[j][i][slicey, slicex],
            linestyle = :dash, color = cmap[i], linewidth = 5)
        
        lines!(axs[j+2], x, itp.(x) - u_fastiso[j][i][slicey, slicex],
            color = cmap[i], label = labels[i], linewidth = 5)
    end
end
Legend(fig[:,3], axs[1])

latexify(x) = ( x, [L"%$xi $\,$" for xi in x] )
etks = latexify(-50:5:50)
utks = latexify(-300:50:50)
rtks = latexify(-3e6:1e6:3e6)

axs[1].title = title1
axs[2].title = title2

axs[1].xticklabelsvisible = false
axs[2].xticklabelsvisible = false
axs[1].xlabelvisible = false
axs[2].xlabelvisible = false
axs[3].xticks = rtks
axs[4].xticks = rtks

axs[2].yticklabelsvisible = false
axs[4].yticklabelsvisible = false
axs[2].ylabelvisible = false
axs[4].ylabelvisible = false
axs[1].yticks = utks
axs[3].yticks = etks

ylims!(axs[1], (-300, 50))
ylims!(axs[2], (-300, 50))
ylims!(axs[3], elims)
ylims!(axs[4], elims)
rowgap!(fig.layout, 80)

figfile = "plots/test3/fastiso3Dgia_heterogeneous=$(heterogeneous)_N=$(N)"
save("$figfile.png", fig)
save("$figfile.pdf", fig)
figfile