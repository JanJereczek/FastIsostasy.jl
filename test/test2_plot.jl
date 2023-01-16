push!(LOAD_PATH, "../")
using FastIsostasy
using CairoMakie
using JLD2
include("helpers_plot.jl")

@inline function main(
    case::String,       # Choose between viscoelastic and purely viscous response.
    n::Int;             # 2^n cells on domain (1)
    make_plot = true,
    make_anim = false,
)

    N = 2^n
    sol_disc = load("data/test2/disc_gpu_N$N.jld2")
    sol_cap = load("data/test2/cap_gpu_N$N.jld2")

    if case == "viscoelastic"
        u_plot = [
            sol_cap["u3D_elastic"] + sol_cap["u3D_viscous"],
            sol_disc["u3D_elastic"] + sol_disc["u3D_viscous"],
        ]
    elseif case == "viscous"
        u_plot = [
            sol_cap["u3D_viscous"],
            sol_disc["u3D_viscous"],
        ]
    end
    
    labels = [
        L"Cap model $\,$",
        L"Disc model $\,$",
    ]

    t_plot_yr = [1.0, 1e3, 5e3, 1e4, 1e5]
    t_plot = years2seconds.(t_plot_yr)

    if make_plot
        response_fig = slice_spada(
            sol_disc["Omega"],
            sol_disc["c"],
            sol_disc["t_vec"],
            t_plot,
            u_plot,
            labels,
            "test2/$(case)_N$N"
        )
    end

    # if make_anim
    #     anim_name = "plots/discload_$(case)_N=$(Omega.N)"
    #     animate_viscous_response(t_vec, Omega, u3D_viscous, anim_name, (-300.0, 50.0))
    # end
end

for case in ["viscoelastic", "viscous"]
    for n in 6:7
        main(case, n)
    end
end