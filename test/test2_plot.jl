push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

@inline function main(
    case::String,       # Choose between viscoelastic and purely viscous response.
    n::Int;             # 2^n cells on domain (1)
    make_anim = false,
    kernel = "gpu",
)

    N = 2^n
    suffix = "$(kernel)_N$N"
    sol_disc = load("data/test2/disc_$suffix.jld2")
    sol_cap = load("data/test2/cap_$suffix.jld2")

    if case == "viscoelastic"
        u_plot = [
            sol_cap["u3D_elastic"] + sol_cap["u3D_viscous"],
            sol_disc["u3D_elastic"] + sol_disc["u3D_viscous"],
            m_per_sec2mm_per_yr.(sol_cap["dudt3D_viscous"]),
            m_per_sec2mm_per_yr.(sol_disc["dudt3D_viscous"]),
        ]
    elseif case == "viscous"
        u_plot = [
            sol_cap["u3D_viscous"],
            sol_disc["u3D_viscous"],
            m_per_sec2mm_per_yr.(sol_cap["dudt3D_viscous"]),
            m_per_sec2mm_per_yr.(sol_disc["dudt3D_viscous"]),
        ]
    end
    
    labels = [
        L"Cap model $\,$",
        L"Disc model $\,$",
        "",
        "",
    ]

    xlabels = [
        "",
        "",
        L"Colatitude $\theta$ (deg)",
        L"Colatitude $\theta$ (deg)",
    ]

    ylabels = [
        L"Total displacement $u$ (m)",
        "",
        L"Displacement rate $\dot{u} \: \mathrm{(mm \, yr^{-1}})$",
        "",
    ]

    t_plot_yr = [1.0, 1e3, 5e3, 1e4, 1e5]
    t_plot = years2seconds.(t_plot_yr)
    response_fig = slice_spada(
        sol_disc["Omega"],
        sol_disc["c"],
        sol_disc["t_out"],
        t_plot,
        u_plot,
        labels,
        xlabels,
        ylabels,
        "test2/$(case)_$suffix"
    )

    # if make_anim
    #     anim_name = "plots/discload_$(case)_N=$(Omega.N)"
    #     animate_viscous_response(t_vec, Omega, u3D_viscous, anim_name, (-300.0, 50.0))
    # end
end

for case in ["viscoelastic", "viscous"]
    for n in 6:6
        main(case, n, kernel = "cpu")
    end
end