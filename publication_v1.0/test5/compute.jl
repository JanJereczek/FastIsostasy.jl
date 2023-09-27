push!(LOAD_PATH, "../")
using FastIsostasy, JLD2
include("../test/helpers/compute.jl")
include("../test/helpers/viscmaps.jl")

function main(; n=5)
    T = Float64
    W = T(3000e3)               # half-length of the square domain (m)
    Omega = ComputationDomain(W, n)
    c = PhysicalConstants()

    lb = [88e3, 180e3, 280e3, 400e3]
    lv = load_wiens2021(Omega)
    p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
    ground_truth = copy(p.effective_viscosity)

    R = T(2000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.(0.0:1_000.0:2_000.0)

    t1 = time()
    results = fastisostasy(t_out, Omega, c, p, Hcylinder, alg=BS3(), interactive_sealevel=false)
    t_fastiso = time() - t1
    println("Took $t_fastiso seconds!")
    println("-------------------------------------")

    tinv = t_out[2:end]
    Hice = [Hcylinder for t in tinv]
    Y = results.u_out[2:end]
    paraminv = InversionProblem(Omega, c, p, tinv, Y, Hice)
    priors, ukiobj = perform(paraminv)
    logeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, paraminv)

    jldsave(
        "../data/test4/n=$n.jld2",
        Omega = Omega, ground_truth = ground_truth, paraminv = paraminv,
        priors = priors, ukiobj = ukiobj,
        logeta = logeta, Gx = Gx, e_mean = e_mean, e_sort = e_sort,
    )
    return nothing
end

main()