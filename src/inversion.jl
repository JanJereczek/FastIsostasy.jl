#=
indices:
params i
samples j
time k
iterations n
=#

"""
    InversionConfig

Struct containing configuration parameters for a [`InversionProblem`].

Need to choose regularization factor α ∈ (0,1],  
When you have enough observation data α=1: no regularization

update_freq 1 : approximate posterior cov matrix with an uninformative prior
            0 : weighted average between posterior cov matrix with an uninformative prior and prior
"""
Base.@kwdef struct InversionConfig
    case::String = "viscosity"
    method::Any = Unscented
    paramspriors::NamedTuple = defaultpriors(case)
    N_iter::Int = 20            # 20
    α_reg::Real = 1.0
    update_freq::Int = 1        # 1
    n_samples::Int = 100        # 100
    scale_obscov::Real = 10000.0
end

"""
    InversionData

Struct containing data (either observational or output of a golden standard model) for a [`InversionProblem`].
"""
struct InversionData{T<:AbstractFloat, M<:Matrix{T}}
    t::Vector{T}        # Time vector
    y::Vector{T}        # Response (anomaly)
    Hice::Vector{M}     # Ice thickness (anomaly)
    idx::BitMatrix      # Index matrix (cell = 1 --> run inversion)
    nt::Int             # number of time steps
    nparams::Int        # number of values to estimate
    nobs::Int           # number of observed values = nt * nparams
end

function InversionData(t, U, Hice, config; Htol::Real = 1.0)   # Hwater
    if (length(t) != length(U)) ||
        (length(t) != length(Hice)) ||
        (length(U) != length(Hice))
        error("The length of the provided a time vector, displacement field history and load history do not match!")
    end

    obs_idx = where_significant(Hice, Htol)
    nt = length(t)
    nparams = sum(obs_idx)
    if config.case == "both"
        nparams *= 2
    end

    nobs = nt * nparams
    y = vcat([u[obs_idx] for u in U]...)
    if nobs != length(y)
        error("The number of observations with significant loading does not correspond to the dimension of the Kalman problem.")
    end
    return InversionData(t, y, Hice, obs_idx, nt, nparams, nobs)
end

"""
    InversionProblem

Struct containing variables and configs for the inversion of
Solid-Earth parameter fields. For now, only viscosity can be inverted but future
versions will support lithosphere rigidity. For now, the unscented Kalman inversion
is the only method available but ensemble Kalman inversion will be available in future.
"""
struct InversionProblem{T<:AbstractFloat, M<:Matrix{T}}
    fip::FastIsoProblem{T, M}
    config::InversionConfig
    data::InversionData{T, M}
    priors::ParameterDistribution
    ukiobj::EnsembleKalmanProcess{T, Int64, Unscented{T, Int64}, DefaultScheduler{T}}
    error::Vector{T}
    out::Vector{Vector{T}}
    G_ens::M
end

function InversionProblem(fip::FastIsoProblem{T, M}, config::InversionConfig,
    data::InversionData{T, M}; saveevery::Int = 1) where {T<:AbstractFloat, M<:Matrix{T}}
    
    # Generating noisy observations
    μ_y = zeros(data.nobs)
    loadscaling_obscov = vcat( [H[data.idx] for H in data.Hice]... )
    Σ_y = uncorrelated_obs_covariance(config.scale_obscov, loadscaling_obscov)
    yn = zeros(T, data.nobs, config.n_samples)
    @inbounds for j in 1:config.n_samples
        yn[:, j] = data.y .+ rand(MvNormal(μ_y, Σ_y))
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

function defaultpriors(case)
    if case == "viscosity"
        return (
            mean = 20.5,
            var = 0.5,
            lowerbound = 19.0,
            upperbound = 22.0,
        )
    elseif case == "rigidity"
        error("Rigidity default choice for inversion not implemeted for now!")
    end
end

"""
    solve!(paraminv::InversionProblem)

Return `priors` and `ukiobj` that allow to extract the results of the parameter
inversion as initialized in `paraminv`.
"""
function solve!(paraminv::InversionProblem{T, M}; verbose::Bool = false) where
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

function forward_fastiso(optimparams::Vector{T}, paraminv::InversionProblem{T, M}) where
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

"""
    where_significant()

Find points of parameter field that can be inverted. We here assume that 
"""
function where_significant(X::Vector{<:Matrix}, tol::Real)
    transientmax = max.( [abs.(x) for x in X]... )
    return transientmax .> tol
end

function uncorrelated_obs_covariance(scale_obscov, loadscaling_obscov)
    diagvar = scale_obscov ./ (loadscaling_obscov .+ 1)   # 10000.0
    return convert(Array, Diagonal(diagvar) )
end

# Actually, this should not be diagonal because there is a correlation between points.
function correlated_obs_covariance()
end

function print_inversion_evolution(paraminv, n, ϕ_n)
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
function extract_inversion(paraminv::InversionProblem)
    p = get_ϕ_mean_final(paraminv.priors, paraminv.ukiobj)
    Gx = get_g_mean_final(paraminv.ukiobj)
    abserror = abs.(paraminv.data.y - Gx)
    return p, Gx, abserror
end