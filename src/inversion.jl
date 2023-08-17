#=
indices:
params i
samples j
time k
iterations n
=#

"""
    InversionConfig

Struct containing configuration parameters for a [`ParamInversion`].
"""
Base.@kwdef struct InversionConfig
    case::String = "viscosity"
    method::Any = Unscented
    paramspriors::NamedTuple = defaultpriors(case)
    N_iter::Int = 20         # 20
    α_reg::Real = 1.0
    update_freq::Int = 1
    n_samples::Int = 20     # 100
    scale_obscov::Real = 10000.0
end

"""
    InversionData

Struct containing data (either observational or output of a golden standard model) for a [`ParamInversion`].
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
    ParamInversion

Struct containing variables and configs for the inversion of
Solid-Earth parameter fields. For now, only viscosity can be inverted but future
versions will support lithosphere rigidity. For now, the unscented Kalman inversion
is the only method available but ensemble Kalman inversion will be available in future.
"""
struct ParamInversion{T<:AbstractFloat, M<:Matrix{T}}
    fip::FastIsoProblem{T, M}
    config::InversionConfig
    data::InversionData{T, M}
    μ_y::Vector{T}
    Σ_y::M
end

function ParamInversion(
    fip::FastIsoProblem{T, M},
    config::InversionConfig,
    data::InversionData{T, M},
) where {T<:AbstractFloat, M<:Matrix{T}}
    # Generating noisy observations
    μ_y = zeros(data.nobs)
    loadscaling_obscov = vcat( [H[data.idx] for H in data.Hice]... )
    Σ_y = uncorrelated_obs_covariance(config.scale_obscov, loadscaling_obscov)
    return ParamInversion(fip, config, data, μ_y, Σ_y)

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
    perform(paraminv::ParamInversion)

Return `priors` and `ukiobj` that allow to extract the results of the parameter
inversion as initialized in `paraminv`.
"""
function perform(paraminv::ParamInversion{T, M}) where {T<:AbstractFloat, M<:Matrix{T}}
    config, data = paraminv.config, paraminv.data
    yn = zeros(T, data.nobs, config.n_samples)
    println("Generating the perturbed ensemble...")
    @inbounds for j in 1:config.n_samples
        yn[:, j] = data.y .+ rand(MvNormal(paraminv.μ_y, paraminv.Σ_y))
    end
    ynoisy = Observations.Observation(yn, paraminv.Σ_y, ["Noisy truth"])

    println("Defining priors...")
    priors = combine_distributions([constrained_gaussian( "p_$(i)",
        config.paramspriors.mean, config.paramspriors.var,
        config.paramspriors.lowerbound, config.paramspriors.upperbound)
        for i in 1:data.nparams])

    println("Inititializing iteration loop...")
    # Here we also could use process = Inversion()
    process = Unscented(mean(priors), cov(priors);
        α_reg = config.α_reg, update_freq = config.update_freq)
    ukiobj = EnsembleKalmanProcess(ynoisy.mean, ynoisy.obs_noise_cov, process)
    err = zeros(T, config.N_iter)
    ϕ_n = get_ϕ_final(priors, ukiobj)       # Params in physical/constrained space
    G_ens = zeros(T, data.nobs, size(ϕ_n, 2))
    
    for n in 1:config.N_iter
        println("Populating G matrix...")
        for j in axes(ϕ_n, 2)
            if rem(j, 10) == 0
                println("n = $n, j = $j")
            end
            G_ens[:, j] .= forward_fastiso(ϕ_n[:, j], paraminv)
        end
        EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
        err[n] = get_error(ukiobj)[end]
        print_kalmanprocess_evolution(paraminv, n, ϕ_n, err[n], ukiobj.process.uu_cov[n])
        ϕ_n .= get_ϕ_final(priors, ukiobj)
    end
    return priors, ukiobj
end

function forward_fastiso(optimparams::Vector, paraminv::ParamInversion)
    dummyfip = paraminv.fip
    remake!(dummyfip)
    config, data = paraminv.config, paraminv.data
    if config.case == "viscosity"
        dummyfip.p.effective_viscosity[data.idx] .= 10.0 .^  optimparams
    elseif config.case == "rigidity"
        dummyfip.p.lithosphere_rigidity[data.idx] .=  optimparams
    elseif config.case == "both"
        dummyfip.p.effective_viscosity[data.idx] .= 10.0 .^  optimparams[1:mparams]
        dummyfip.p.lithosphere_rigidity[data.idx] .=  optimparams[mparams+1:end]
    end
    solve!(dummyfip)
    # results taken from k=2 onwards because k=1 returns the solution at time t=0.
    return vcat([reshape(u[data.idx], data.nparams) for u in  dummyfip.out.u[2:end]]...)
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

function print_kalmanprocess_evolution(paraminv, n, ϕ_n, err_n, cov_n)
    println("------------------")
    println("size: $(size(ϕ_n))")
    if paraminv.config.case == "viscosity" || paraminv.config.case == "rigidity"
        println("mean $(paraminv.config.case): $(mean(ϕ_n))")
    elseif paraminv.config.case == "both"
        m = paraminv.data.nparams ÷ 2
        meanvisc = round( mean(ϕ_n[1:m, :]), digits = 4)
        meanrigd = round( mean(ϕ_n[m+1:end, :]), digits = 4)
        println("mean viscosity: $meanvisc,  mean rigidity: $meanrigd")
    end
    println("Iteration: $n, Error: $err_n, norm(Cov): $(norm(cov_n))")
    return nothing
end

"""
    extract_inversion()

Extract results of parameter inversion from the `priors` and `ukiobj` that
resulted from `perform!(paraminv::ParamInversion)`.
"""
function extract_inversion(priors, ukiobj, data::InversionData)
    p = get_ϕ_mean_final(priors, ukiobj)
    Gx = get_g_mean_final(ukiobj)
    e_mean = mean(abs.(data.y - Gx))
    e_sort = sort(abs.(data.y - Gx))
    return p, Gx, e_mean, e_sort
end