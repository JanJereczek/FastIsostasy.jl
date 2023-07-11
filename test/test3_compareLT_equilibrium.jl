push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2, DelimitedFiles
include("helpers_plot.jl")

global include_elastic = true

n = 8
N = 2^n
kernel = "cpu"
suffix = "$(kernel)_Nx$(N)_Ny$(N)_dense"

function get_denseoutput_fastiso(suffix)
    sol_lo_D = load("data/test3/gaussian_lo_D_$suffix.jld2")
    sol_hi_D = load("data/test3/gaussian_hi_D_$suffix.jld2")
    sol_no_D = load("data/test3/gaussian_no_D_$suffix.jld2")
    results = [sol["results"] for sol in [sol_lo_D, sol_hi_D, sol_no_D]]
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

fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel = L"Position along great circle (m) $\,$",
    ylabel = L"Vertical displacement (m) $\,$",
)
labels = [L"Thinning lithosphere $\,$", L"Thickening lithosphere $\,$", L"No lithosphere $\,$"]
for i in eachindex(u_fastiso)
    lines!(ax, x, u_fastiso[i][end][slicey, slicex], label = labels[i])
end

Legend(fig[:,2], ax)

figfile = "plots/test3/LTequilibriumcompare_elastic=$(include_elastic)_N=$(N)"
save("$figfile.png", fig)
save("$figfile.pdf", fig)