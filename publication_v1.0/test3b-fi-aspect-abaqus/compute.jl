using FastIsostasy
using JLD2, LinearAlgebra
include("../../test/helpers/compute.jl")

function main(n, earthtype, timescale)
    println("----------------------------------------")
    println("Computing $timescale experiment on $earthtype Earth")
    W = 1500e3
    Omega = ComputationDomain(W, n, use_cuda = false)
    # c = PhysicalConstants(rho_litho = 3.037e3, rho_uppermantle = 3.438e3, rho_ice = 0.931e3)
    c = PhysicalConstants(rho_litho = 4.492e3, rho_uppermantle = 4.492e3, rho_ice = 0.931e3)

    if earthtype == "homogeneous"
        p = LayeredEarth(Omega, layer_boundaries = [70e3, 420e3], layer_viscosities = [1e21, 1e21])
    elseif earthtype == "heterogeneous"
        lvz = (Omega.R .< 100e3) .* 1e19
        lvz[lvz .< 1] .= 1e21
        nvz = fill(1e21, size(lvz)...)
        p = LayeredEarth(Omega, layer_boundaries = [70e3, 170e3],
            layer_viscosities = cat(lvz, nvz, dims=3))
    end

    R = 100e3
    H = 100.0
    Hice_max = uniform_ice_cylinder(Omega, R, H)

    if timescale == "short"
        tvec = collect(0.0:10.0:200.0)
        dHdt = Hice_max ./ 100.0
        Hice = [min.(dHdt .* t, H) for t in tvec]
    elseif timescale == "long"
        dt = 1e3
        t1, t2, t3, t4 = 0, 90e3, 100e3, 110e3
        tvec1 = collect(t1:1e3:t2)
        tvec2 = collect(t2+dt:1e3:t3)
        tvec3 = collect(t3+dt:1e3:t4)
        dHdt1 = Hice_max ./ t2
        dHdt2 = Hice_max ./ (t3-t2)
        Hice1 = [min.(dHdt1 .* t, H) for t in tvec1]
        Hice2 = [Hice_max - min.(dHdt2 .* (t - t2), H) for t in tvec2]
        Hice3 = [zeros(size(Hice_max)...) for t in tvec3]
        tvec = vcat(tvec1, tvec2, tvec3)
        Hice = vcat(Hice1, Hice2, Hice3)
    end

    tsecvec = years2seconds.(tvec)
    fip = FastIsoProblem(Omega, c, p, tsecvec, false, tsecvec, Hice)
    solve!(fip)
    println("Took $(fip.out.computation_time) seconds!")
    println("----------------------------------------")

    filename = "../data/test3b/$earthtype-$timescale-N$(Omega.Nx)"
    savefip("$filename.nc", fip)
    @save "$filename.jld2" fip Hice
end

for earthtype in ["heterogeneous"]
    for timescale in ["short"]
        main(8, earthtype, timescale)
    end
end