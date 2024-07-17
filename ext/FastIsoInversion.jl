module FastIsoInversion

using FastIsostasy
using Distributions
using EnsembleKalmanProcesses: EnsembleKalmanProcesses, EnsembleKalmanProcess,
                               Unscented, get_error, get_g_mean_final,
                               get_ϕ_final, get_ϕ_mean_final
using EnsembleKalmanProcesses.Observations: Observations
using EnsembleKalmanProcesses.ParameterDistributions: ParameterDistributions,
                                                      ParameterDistribution,
                                                      combine_distributions,
                                                      constrained_gaussian
using LinearAlgebra

#=
indices:
params i
samples j
time k
iterations n
=#


function FastIsostasy.inversion_problem(fip, config, data; saveevery::Int = 1)
    T = Float64
    # Generating noisy observations
    μ_y = zeros(data.nobs)
    loadscaling_obscov = vcat( [H[data.idx] for H in data.Hice]... )
    Σ_y = uncorrelated_obs_covariance(config.scale_obscov, loadscaling_obscov)
    yn = zeros(T, data.nobs, config.n_samples)
    @inbounds for j in 1:config.n_samples
        yn[:, j] = data.y .+ rand(Distributions.MvNormal(μ_y, Σ_y))
    end
    ynoisy = Observations.Observation(yn, Σ_y, ["Noisy truth"])

    # Defining priors
    priors = combine_distributions([constrained_gaussian( "p_$(i)",
        config.paramspriors.mean, config.paramspriors.var,
        config.paramspriors.lowerbound, config.paramspriors.upperbound)
        for i in 1:data.nparams])

    # Init process and arrays
    process = Unscented(mean(priors), cov(priors);  # Could also use process = Inversion()
        α_reg = config.α_reg, update_freq = config.update_freq)
    ukiobj = EnsembleKalmanProcess(ynoisy.mean, ynoisy.obs_noise_cov, process)
    error = fill(T(Inf), config.N_iter)
    out = [fill(Inf, data.nparams) for _ in 1:saveevery:config.N_iter+1]

    ϕ_tool = get_ϕ_final(priors, ukiobj)       # Params in physical/constrained space
    G_ens = zeros(T, data.nobs, size(ϕ_tool, 2))

    return InversionProblem(fip, config, data, priors, ukiobj, error, out, G_ens)

end

function uncorrelated_obs_covariance(scale_obscov, loadscaling_obscov)
    diagvar = scale_obscov ./ (loadscaling_obscov .+ 1)   # 10000.0
    return convert(Array, Diagonal(diagvar) )
end

"""
    solve!(paraminv::InversionProblem)

Return `priors` and `ukiobj` that allow to extract the results of the parameter
inversion as initialized in `paraminv`.
"""
function FastIsostasy.solve!(paraminv::InversionProblem{T, M}; verbose::Bool = false) where
    {T<:AbstractFloat, M<:Matrix{T}}

    paraminv.out[1] .= get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)

    for n in 1:paraminv.config.N_iter

        # Get params in physical/constrained space
        ϕ_n = get_ϕ_final(paraminv.priors, paraminv.ukiobj)

        @inbounds for j in axes(ϕ_n, 2)
            if verbose && (rem(j, 10) == 0)
                println("Populating ensemble displacement matrix at n = $n, j = $j")
            end
            paraminv.G_ens[:, j] = forward_fastiso(ϕ_n[:, j], paraminv)
        end

        # @threads for j in axes(ϕ_n, 2)
        #     paraminv.G_ens[:, j] = forward_fastiso(ϕ_n[:, j], paraminv)
        # end

        if verbose
            println("Extrema of ensemble displacement matrix: $(extrema(paraminv.G_ens))")
        end

        EnsembleKalmanProcesses.update_ensemble!(paraminv.ukiobj, paraminv.G_ens)
        paraminv.error[n] = get_error(paraminv.ukiobj)[end]
        paraminv.out[n+1] .= get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
        print_inversion_evolution(paraminv, n, ϕ_n)
    end
    return nothing
end

function FastIsostasy.forward_fastiso(optimparams::Vector{T}, paraminv::InversionProblem{T, M}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    remake!(paraminv.fip)
    config, data = paraminv.config, paraminv.data
    if config.case == "viscosity"
        paraminv.fip.p.effective_viscosity[data.idx] .= 10.0 .^  optimparams
    elseif config.case == "rigidity"
        paraminv.fip.p.lithosphere_rigidity[data.idx] .=  optimparams
    elseif config.case == "both"
        paraminv.fip.p.effective_viscosity[data.idx] .= 10.0 .^  optimparams[1:mparams]
        paraminv.fip.p.lithosphere_rigidity[data.idx] .=  optimparams[mparams+1:end]
    end
    solve!(paraminv.fip)
    # results taken from k=2 onwards because k=1 returns the solution at time t=0.
    return vcat([reshape(u[data.idx], data.nparams) for u in  paraminv.fip.out.u[2:end]]...)
end

# Actually, this should not be diagonal because there is a correlation between points.
function correlated_obs_covariance()
end

function FastIsostasy.print_inversion_evolution(paraminv, n, ϕ_n)
    err_n = paraminv.error[n]
    cov_n = paraminv.ukiobj.process.uu_cov[n]

    println("------------------")
    println("Ensemble size: $(size(ϕ_n))")
    println("Iteration: $n, Error: $err_n, norm(Cov): $(norm(cov_n))")
    if paraminv.config.case == "viscosity" || paraminv.config.case == "rigidity"
        println("Mean $(paraminv.config.case): $(round(mean(ϕ_n), digits = 4))")
        println("Extrema of $(paraminv.config.case): $(extrema(ϕ_n))")
    elseif paraminv.config.case == "both"
        m = paraminv.data.nparams ÷ 2
        meanvisc = round( mean(ϕ_n[1:m, :]), digits = 4)
        meanrigd = round( mean(ϕ_n[m+1:end, :]), digits = 4)
        println("Mean viscosity: $meanvisc,  mean rigidity: $meanrigd")
    end
    return nothing
end

"""
    extract_inversion()

Extract results of parameter inversion from the `priors` and `ukiobj` that
resulted from `solve!(paraminv::InversionProblem)`.
"""
function FastIsostasy.extract_inversion(paraminv::InversionProblem)
    p = get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
    Gx = get_g_mean_final(paraminv.ukiobj)
    abserror = abs.(paraminv.data.y - Gx)
    return p, Gx, abserror
end

FastIsostasy.testfunc(x) = println("Externalised inversion is imported correctly.")

end