push!(LOAD_PATH, "../")
using FastIsostasy, JLD2
include("../../test/helpers/compute.jl")
include("../../test/helpers/viscmaps.jl")

function main(; n=5)
    Omega = ComputationDomain(3000e3, 5)
    c = PhysicalConstants()
    lb = [88e3, 180e3, 280e3, 400e3]
    lv = load_wiens2021(Omega)
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    R, H = 1000e3, 1e3
    Hice = uniform_ice_cylinder(Omega, R, H, center = [-1000e3, -1000e3])
    t_out = years2seconds.(1e3:1e3:2e3)
    true_viscosity = copy(p.effective_viscosity)
    fip = FastIsoProblem(Omega, c, p, t_out, false, Hice)
    solve!(fip)

    config = InversionConfig(N_iter = 15)
    data = InversionData(copy(fip.out.t[2:end]), copy(fip.out.u[2:end]), copy([Hice, Hice]), config)
    paraminv = InversionProblem(deepcopy(fip), config, data)

    init_viscosity = copy(true_viscosity)
    init_viscosity[paraminv.data.idx] .= 10 .^ get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
    solve!(paraminv)
    final_viscosity = copy(true_viscosity)
    final_viscosity[paraminv.data.idx] .= 10 .^ get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)

    @save "../data/test4/n=$n.jld2" fip paraminv init_viscosity final_viscosity
    return nothing
end

main()