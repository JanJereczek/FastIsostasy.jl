using CairoMakie
using JLD2
include("helpers_plots.jl")

@inline function main(
    n::Int;             # 2^n cells on domain (1)
    make_plot = true,
    make_anim = false,
)

    N = 2^n
    sol_disc = load("data/test2_disc_N$N.jld2")
    sol_cap = load("data/test2_cap_N$N.jld2")

    u_plot = [
        sol_cap["u3D_elastic"] + sol_cap["u3D_viscous"],
        sol_disc["u3D_elastic"] + sol_disc["u3D_viscous"],
    ]
    labels = [
        L"Cap model $\,$",
        L"Disc model $\,$",
    ]

    if make_plot
        response_fig = slice_response(
            sol_disc["Omega"],
            sol_disc["c"],
            sol_disc["t_vec"],
            years2seconds.([0.0, 1e3, 2e3, 5e3, 1e4, 1e5]),
            u_plot,
            labels,
            "test2_N$N"
        )
    end

    # if make_anim
    #     anim_name = "plots/discload_$(case)_N=$(Omega.N)"
    #     animate_viscous_response(t_vec, Omega, u3D_viscous, anim_name, (-300.0, 50.0))
    # end
end

for n in 7:7
    main(n)
end
