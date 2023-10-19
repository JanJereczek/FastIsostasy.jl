push!(LOAD_PATH, "../")
using FastIsostasy, JLD2
include("../../test/helpers/compute.jl")

function main(n, N_iter)
    Omega = ComputationDomain(3000e3, n)
    c = PhysicalConstants()
    lb = [88e3, 180e3, 280e3, 350e3]
    dims, eta, eta_itp = load_wiens2021(Omega)
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = 10 .^ eta)
    R, H = 1000e3, 1e3
    Hice = uniform_ice_cylinder(Omega, R, H, center = [-1000e3, -1000e3])
    t_out = years2seconds.(1e3:1e3:2e3)
    true_viscosity = copy(p.effective_viscosity)
    fip = FastIsoProblem(Omega, c, p, t_out, false, Hice)
    solve!(fip)

    config = InversionConfig(N_iter = N_iter)
    data = InversionData(copy(fip.out.t[2:end]), copy(fip.out.u[2:end]), copy([Hice, Hice]), config)
    paraminv = InversionProblem(deepcopy(fip), config, data)

    init_viscosity = copy(true_viscosity)
    init_viscosity[paraminv.data.idx] .= 10 .^ get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
    solve!(paraminv)
    final_viscosity = copy(true_viscosity)
    final_viscosity[paraminv.data.idx] .= 10 .^ get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)

    # @save "../data/test5/n=$n.jld2" fip paraminv init_viscosity final_viscosity
    return nothing
end

@time main(5, 1)

#=
@time main(5, 1)
9.774572 seconds (10.88 M allocations: 14.960 GiB, 15.09% gc time)
=#