#####################################################
# Ensemble Kalman processes
#####################################################

using JLD2, FastIsostasy
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
using Distributions, LinearAlgebra, Interpolations
using CairoMakie

include("../test/external_viscosity_maps.jl")
include("../test/helpers_compute.jl")

function get_wiens_layervisc(Omega)
    halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)
    Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    return lv
end

### n-dim case ###
function main(n)
    T = Float64
    L = T(3000e3)
    Omega = ComputationDomain(L, n)
    c = PhysicalConstants()

    lb = [88e3, 180e3, 280e3, 400e3]
    lv = get_wiens_layervisc(Omega)
    p = MultilayerEarth(
        Omega,
        c,
        layers_begin = lb,
        layers_viscosity = lv,
    )
    ground_truth = copy(p.effective_viscosity)
    R = T(2000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    forced_idx = Hcylinder .> 1
    nparams = sum(forced_idx)

    logeta_meanprior = 20.5
    logeta_varprior = 0.5
    logeta_lb = 19.0
    logeta_ub = 22.0
    priors = combine_distributions([constrained_gaussian( "logeta_$(i)",
        logeta_meanprior, logeta_varprior, logeta_lb, logeta_ub) for i in 1:nparams])

    N_iter = 20         # number of iterations
    α_reg = 1.0         # regularization parameter
    update_freq = 1

    # t_out = years2seconds.([0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    t_out = years2seconds.(0.0:1_000.0:2_000.0)
    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3())

    function fastiso(logeta)
        # kalmanlogeta = fill(logeta[1], Omega.N, Omega.N)
        # kalmanlogeta[forced_idx] .= logeta[2:end]
        p.effective_viscosity[forced_idx] = 10 .^ logeta
        results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3())
        return vcat([reshape(results.viscous[j][forced_idx], sum(forced_idx))
            for j in eachindex(results.viscous)[2:end]]...)
    end

    U = vcat([reshape(results.viscous[j][forced_idx], sum(forced_idx))
        for j in eachindex(results.viscous)[2:end]]...)
    n_samples = 100
    y_t = zeros(length(U), n_samples)
    covH = reshape( 10000.0 ./ (Hcylinder[forced_idx] .+ 1), sum(forced_idx) )

    # Actually, this should not be diagonal because there is a correlation between points.
    Γy = convert(Array, Diagonal(vcat([covH for j in eachindex(results.viscous)[2:end]]...)) )
    μ = zeros(length(U))

    for i in 1:n_samples
        y_t[:, i] = U .+ rand(MvNormal(μ, Γy))
    end

    truth = Observations.Observation(y_t, Γy, ["Wiens"])
    truth_sample = truth.mean

    # Here we also could use process = Inversion()
    process = Unscented(mean(priors), cov(priors); α_reg = α_reg, update_freq = update_freq)
    ukiobj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)

    err = zeros(N_iter)
    for n in 1:N_iter
        # Return transformed parameters in physical/constrained space
        ϕ_n = get_ϕ_final(priors, ukiobj)
        # Evaluate forward map
        println("size: ", size(ϕ_n), ",  mean viscosity: $(mean(ϕ_n))")
        G_n = [fastiso(ϕ_n[:, i]) for i in 1:size(ϕ_n)[2]]
        G_ens = hcat(G_n...)  # reformat
        EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
        err[n] = get_error(ukiobj)[end]
        println(
            "Iteration: " * string(n) *
            ", Error: " * string(err[n]) *
            ", norm(Cov):" * string(norm(ukiobj.process.uu_cov[n])),
        )
    end
    ϕ_n = get_ϕ_final(priors, ukiobj)
    return forced_idx, ground_truth, priors, ukiobj, p, U
end

n = 5
tbegin = time()
forced_idx, y, priors, ukiobj, p, U = main(n)
runtime = round(time() - tbegin)
println("UKI took $runtime seconds to run!")

logeta = get_ϕ_mean_final(priors, ukiobj)
ufwd = get_g_mean_final(ukiobj)
e_mean = mean(abs.(U - ufwd))
e_sort = sort(abs.(U - ufwd))

titles = [L"True viscosity field $\,$", L"Estimated viscosity field $\,$"]
cmap = cgrad(:jet, rev=true)
ncols = length(titles)
fig = Figure(resolution = (1600, 900), fontsize = 28)
axs = [Axis(fig[1, j], aspect = DataAspect(), title=titles[j]) for j in 1:ncols]
[hidedecorations!(axs[j]) for j in 1:ncols]

x = 1:2^n
heatmap!(axs[1], x, x, log10.(y), colorrange = (20, 21), colormap = cmap)
contour!(axs[1], x .+ 0.5, x .+ 0.5, forced_idx, levels = [0.99], color = :white, linewidth = 5)

heatmap!(axs[2], x, x, log10.(p.effective_viscosity), colorrange = (20, 21), colormap = cmap)
contour!(axs[2], x .+ 0.5, x .+ 0.5, forced_idx, levels = [0.99], color = :white, linewidth = 5)

Colorbar(fig[2, :], colorrange = (20, 21), colormap = cmap, vertical = false, width = Relative(0.5))
fig