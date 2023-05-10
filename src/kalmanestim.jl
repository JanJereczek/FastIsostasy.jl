#####################################################
# Ensemble Kalman processes
#####################################################

using JLD2, FastIsostasy
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
using Distributions
using LinearAlgebra
using Interpolations

include("../test/external_viscosity_maps.jl")
include("../test/helpers_compute.jl")

### scalar case ###
n = 6
T = Float64
L = T(3000e3)
Omega = ComputationDomain(L, n)
c = PhysicalConstants()
p = MultilayerEarth(Omega, c)

logeta_meanprior = 20.0
logeta_varprior = 0.5
logeta_lb = 18.0
logeta_ub = 23.0
priors = combine_distributions([constrained_gaussian("logeta",
    logeta_meanprior, logeta_varprior, logeta_lb, logeta_ub)])

N_iter = 50         # number of iterations
α_reg = 1.0         # regularization parameter
update_freq = 1

R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.(0.0:1_000.0:1_000.0)
results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3())

function fastiso(logeta)
    p.effective_viscosity = 10.0 .^ fill(logeta[1], Omega.N, Omega.N)
    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3())
    return reshape(results.viscous[end], Omega.N*Omega.N)
end

Hice = [Hcylinder for t in t_out]
U = reshape(results.viscous[end], Omega.N*Omega.N)
n_samples = 100
y_t = zeros(length(U), n_samples)
Γy = convert(Array, Diagonal(ones(Omega.N*Omega.N)))
μ = zeros(length(U))

for i in 1:n_samples
    y_t[:, i] = U .+ rand(MvNormal(μ, Γy))
end

truth = Observations.Observation(y_t, Γy, ["Wiens"])
truth_sample = truth.mean

process = Unscented(mean(priors), cov(priors); α_reg = α_reg, update_freq = update_freq)
ukiobj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)

err = zeros(N_iter)
for n in 1:N_iter
    # Return transformed parameters in physical/constrained space
    ϕ_n = get_ϕ_final(priors, ukiobj)
    # Evaluate forward map
    println("size: ", size(ϕ_n))
    G_n = [fastiso(ϕ_n[:, i]) for i in 1:size(ϕ_n)[2]]
    G_ens = hcat(G_n...)  # reformat
    EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
    err[n] = get_error(ukiobj)[end]
    println(
        "Iteration: " * string(n) *
        ", Error: " * string(err[n]) *
        " norm(Cov):" * string(norm(ukiobj.process.uu_cov[n])),
    )
end

### n-dim case ###
function main()
    n = 4
    T = Float64
    L = T(3000e3)
    Omega = ComputationDomain(L, n)
    c = PhysicalConstants()

    lb = [88e3, 180e3, 280e3, 400e3]
    halfspace_logviscosity = fill(21.0, Omega.N, Omega.N)
    Eta, Eta_mean, z = load_wiens_2021(Omega.X, Omega.Y)
    eta_interpolators, eta_mean_interpolator = interpolate_viscosity_xy(
        Omega.X, Omega.Y, Eta, Eta_mean)
    lv = 10.0 .^ cat(
        [itp.(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        # [eta_interpolators[1].(Omega.X, Omega.Y) for itp in eta_interpolators]...,
        halfspace_logviscosity,
        dims=3,
    )
    p = MultilayerEarth(
        Omega,
        c,
        layers_begin = lb,
        layers_viscosity = lv,
    )

    logeta_meanprior = 20.5
    logeta_varprior = 0.5
    logeta_lb = 19.0
    logeta_ub = 22.0
    priors = combine_distributions([constrained_gaussian( "logeta_$(i)",
        logeta_meanprior, logeta_varprior, logeta_lb, logeta_ub) for i in 1:Omega.N*Omega.N])

    N_iter = 50         # number of iterations
    α_reg = 0.9         # regularization parameter
    update_freq = 1

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 1000.0])
    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3())

    function fastiso(logeta)
        p.effective_viscosity = 10 .^ reshape(logeta, Omega.N, Omega.N)
        results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3())
        return reshape(results.viscous[end], Omega.N*Omega.N)
    end

    Hice = [Hcylinder for t in t_out]
    U = reshape(results.viscous[end], Omega.N*Omega.N)
    n_samples = 100
    y_t = zeros(length(U), n_samples)
    Γy = convert(Array, Diagonal(ones(Omega.N*Omega.N)))
    μ = zeros(length(U))

    for i in 1:n_samples
        y_t[:, i] = U .+ rand(MvNormal(μ, Γy))
    end

    truth = Observations.Observation(y_t, Γy, ["Wiens"])
    truth_sample = truth.mean

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
    return ϕ_n
end

ϕ_n = main()
# # UKI results: the mean is in ukiobj.process.u_mean
# #              the covariance matrix is in ukiobj.process.uu_cov
# θvec_true = transform_constrained_to_unconstrained(priors, ϕ_true)

# println("True parameters (transformed): ")
# println(θvec_true)

# println("\nUKI results:")
# println(get_u_mean_final(ukiobj))

# u_stored = get_u(ukiobj, return_array = false)
# g_stored = get_g(ukiobj, return_array = false)
# @save data_save_directory * "parameter_storage_uki.jld2" u_stored
# @save data_save_directory * "data_storage_uki.jld2" g_stored


### n-dim case with tsvd ###

using CairoMakie, TSVD
heatmap(p.effective_viscosity)
U, s, V = tsvd(p.effective_viscosity, 10)
heatmap(U * Diagonal(s) * V')