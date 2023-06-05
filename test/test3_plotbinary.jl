push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

function main(
    n::Int;             # 2^n cells on domain (1)
    make_anim = false,
    kernel = "gpu",
)

    N = 2^n
    suffix = "$(kernel)_N$N"
    sol_D = load("data/test3/binaryD_$suffix.jld2")
    sol_η = load("data/test3/binaryη_$suffix.jld2")
    sol_Dη = load("data/test3/binaryDη_$suffix.jld2")

    u_plot = [sol["u3D_viscous"] for sol in [sol_D, sol_η, sol_Dη]]
    dudt_plot = [m_per_sec2mm_per_yr.(sol["dudt3D_viscous"]) for 
                sol in [sol_D, sol_η, sol_Dη]]
    var_plot = vcat(u_plot, dudt_plot)
    
    labels = [
        W"Binary rigidity $D$",
        W"Binary channel viscosity $\eta_\mathrm{1}$",
        W"Binary rigidity $D$ and channel viscosity $\eta_\mathrm{1}$",
        "",
        "",
        "",
    ]

    xlabels = [
        W"Colatitude $\theta$ (°)",
        W"Colatitude $\theta$ (°)",
        W"Colatitude $\theta$ (°)",
        # "",
        # "",
        # "",
        W"Colatitude $\theta$ (deg)",
        W"Colatitude $\theta$ (deg)",
        W"Colatitude $\theta$ (deg)",
    ]

    ylabels = [
        W"Total displacement $u$ (m)",
        "",
        "",
        W"Displacement rate $\dot{u} \: \mathrm{(mm \, yr^{-1}})$",
        "",
        "",
    ]

    t_plot_yr = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
    t_plot = years2seconds.(t_plot_yr)
    response_fig = slice_test3(
        sol_D["Omega"],
        sol_D["c"],
        sol_D["t_out"],
        t_plot,
        u_plot, # var_plot,
        labels,
        xlabels,
        ylabels,
        "test3/$suffix"
    )

end

for n in 6:6
    main(n, kernel = "cpu")
end