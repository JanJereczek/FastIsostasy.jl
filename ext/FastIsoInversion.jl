module FastIsoInversion

using FastIsostasy
using Distributions
using EnsembleKalmanProcesses:
    EnsembleKalmanProcesses,
    EnsembleKalmanProcess,
    Unscented,
    get_error,
    get_g_mean_final,
    get_ϕ_final,
    get_ϕ_mean_final
using EnsembleKalmanProcesses.Observations: Observations
using EnsembleKalmanProcesses.ParameterDistributions:
    ParameterDistributions,
    ParameterDistribution,
    combine_distributions,
    constrained_gaussian
using LinearAlgebra
using .Threads

"""
    inversion_problem(fip, config, data, reduction, priors; save_stride_iter::Int = 1)

Return an `InversionProblem` struct that contains the variables and configurations
for the inversion of solid-Earth parameters. The data you are inverting for is
defined by `reduction<:ParameterReduction` and the behaviour of
`reconstruct!(fip, reduction)`.


# Indices internally used for inversion
- params: `i`
- samples: `j`
- time: `k`
- iterations: `n`

"""
function FastIsostasy.inversion_problem(
    fip::FastIsoProblem{T,L,M,C,FP,IP,O},
    config::InversionConfig,
    data::InversionData{T,M},
    reduction::R,
    priors;
    save_stride_iter = 1,
) where {T<:AbstractFloat,L,M<:Matrix{T},C,FP,IP,O,R<:ParameterReduction{T}}


    println("Generating noisy data set...")
    Σ_y = uncorrelated_obs_covariance(
        config.scale_obscov,
        ones(T, data.countmask * data.nt),
    )
    yn = zeros(T, data.countmask * data.nt, config.n_samples)
    y = vcat([yy[data.mask] for yy in data.Y]...)
    @inbounds for j in axes(yn, 2)
        view(yn, :, j) .=
            y .+ rand(Distributions.MvNormal(zeros(T, data.countmask * data.nt), Σ_y))
    end

    println("Observing data...")
    ynoisy = Observations.Observation(yn, Σ_y, ["Noisy truth"])

    println("Defining process ...")
    process = Unscented(
        mean(priors),
        cov(priors);  # Could also use process = Inversion()
        α_reg = config.α_reg,
        update_freq = config.update_freq,
    )
    ukiobj = EnsembleKalmanProcess(ynoisy.mean, ynoisy.obs_noise_cov, process)

    println("Initializing arrays...")
    error = fill(T(Inf), config.N_iter)
    out = [fill(Inf, reduction.nparams) for _ = 1:save_stride_iter:config.N_iter+1]
    ϕ_tool = get_ϕ_final(priors, ukiobj)       # Params in physical/constrained space
    G_ens = zeros(T, data.countmask * data.nt, size(ϕ_tool, 2))

    return InversionProblem(
        fip,
        config,
        data,
        reduction,
        priors,
        ukiobj,
        error,
        out,
        G_ens,
    )

end

function uncorrelated_obs_covariance(scale_obscov, loadscaling_obscov)
    diagvar = scale_obscov ./ (loadscaling_obscov .+ 1)   # 10000.0
    return convert(Array, Diagonal(diagvar))
end

"""
    solve!(paraminv::InversionProblem)

Return `priors` and `ukiobj` that allow to extract the results of the parameter
inversion as initialized in `paraminv`.
"""
function FastIsostasy.solve!(paraminv::InversionProblem; verbose::Bool = false)

    paraminv.out[1] .= get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
    ϕ_n = get_ϕ_final(paraminv.priors, paraminv.ukiobj)
    fips = [deepcopy(paraminv.fip) for _ = 1:nthreads()]

    for n = 1:paraminv.config.N_iter

        # Get params in physical/constrained space
        ϕ_n .= get_ϕ_final(paraminv.priors, paraminv.ukiobj)

        Threads.@threads for j in axes(ϕ_n, 2)
            id = Threads.threadid()
            if verbose # && (rem(j, 10) == 0)
                println("Populating ̂Y at: thread = $id,  n = $n,  j = $j")
            end
            FastIsostasy.reconstruct!(fips[id], ϕ_n[:, j], paraminv.reduction)
            paraminv.G_ens[:, j] .=
                forward_fastiso(fips[id], paraminv.reduction, paraminv.data)
        end

        if verbose
            println("Extrema of ̂Y: $(extrema(paraminv.G_ens))")
        end

        EnsembleKalmanProcesses.update_ensemble!(paraminv.ukiobj, paraminv.G_ens)
        paraminv.error[n] = get_error(paraminv.ukiobj)[end]
        paraminv.out[n+1] .= get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
        print_inversion_evolution(paraminv, n, ϕ_n, paraminv.reduction)
    end

    FastIsostasy.reconstruct!(
        paraminv.fip,
        get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj),
        paraminv.reduction,
    )

    return nothing
end

function FastIsostasy.forward_fastiso(
    fip::FastIsoProblem,
    r::ParameterReduction,
    d::InversionData,
)
    remake!(fip)
    solve!(fip)
    return FastIsostasy.extract_output(fip, r, d)
end

# Actually, this should not be diagonal because there is a correlation between points.
function correlated_obs_covariance() end

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