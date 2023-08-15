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
    N_iter::Int = 20
    α_reg::Real = 1.0
    update_freq::Int = 1
    n_samples::Int = 100
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
    Omega::ComputationDomain{T, M}
    c::PhysicalConstants{T}
    p::LateralVariability{T, M}
    config::InversionConfig
    data::InversionData{T, M}
    μ_y::Vector{T}
    Σ_y::M
end

function ParamInversion(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    config::InversionConfig,
    data::InversionData{T, M},
) where {T<:AbstractFloat, M<:Matrix{T}}
    # Generating noisy observations
    μ_y = zeros(data.nobs)
    loadscaling_obscov = vcat( [H[data.idx] for H in data.Hice]... )
    Σ_y = uncorrelated_obs_covariance(scale_obscov, loadscaling_obscov)
    return ParamInversion(Omega, c, p, config, data, μ_y, Σ_y)

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
function perform(paraminv::ParamInversion)

    c, d = paraminv.config, paraminv.data
    yn = zeros(d.nobs, c.n_samples)
    @inbounds for j in 1:c.n_samples
        yn[:, j] = d.y .+ rand(MvNormal(paraminv.μ_y, paraminv.Σ_y))
    end
    ynoisy = Observations.Observation(yn, paraminv.Σ_y, ["Noisy truth"])

    priors = combine_distributions([constrained_gaussian( "p_$(i)",
        c.paramspriors.mean, c.paramspriors.var,
        c.paramspriors.lowerbound, c.paramspriors.upperbound)
        for i in 1:c.nparams])

    # Here we also could use process = Inversion()
    process = Unscented(mean(priors), cov(priors);
        α_reg = c.α_reg, update_freq = c.update_freq)
    ukiobj = EnsembleKalmanProcess(ynoisy.mean, ynoisy.obs_noise_cov, process)
    err = zeros(c.N_iter)
    for n in 1:c.N_iter
        ϕ_n = get_ϕ_final(priors, ukiobj)       # Params in physical/constrained space
        G_n = [forward_fastiso(ϕ_n[:, j], paraminv) for j in axes(ϕ_n, 2)]      # Evaluate forward map
        G_ens = hcat(G_n...)
        EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
        err[n] = get_error(ukiobj)[end]
        print_kalmanprocess_evolution(paraminv, n, ϕ_n, err[n], ukiobj.process.uu_cov[n])
    end
    return priors, ukiobj
end

# ϕ_n = get_ϕ_final(priors, ukiobj)
function FastIsoProblem(params::Vector, paraminv::ParamInversion)
    config, data = paraminv.data, paraminv.config
    if config.case == "viscosity"
        paraminv.p.effective_viscosity[data.idx] .= 10 .^ params
    elseif config.case == "rigidity"
        paraminv.p.lithosphere_rigidity[data.idx] .= params
    elseif config.case == "both"
        paraminv.p.effective_viscosity[data.idx] .= 10 .^ params[1:mparams]
        paraminv.p.lithosphere_rigidity[data.idx] .= params[mparams+1:end]
    end
    interactive_sealevel = false
    return FastIsoProblem(paraminv.Omega, paraminv.c, paraminv.p, paraminv.data.t,
        interactive_sealevel, paraminv.data.Hice, verbose = false)
end

function forward_fastiso(params::Vector, paraminv::ParamInversion)
    fip = FastIsoProblem(params, paraminv)
    solve!(fip)
    Gx = vcat([reshape(u[paraminv.data.idx], paraminv.nparams) for u in fip.out.u[2:end]]...)
    # results taken from k=2 onwards because k=1 returns the solution at time t=0.
    return Gx
end

"""
    where_significant()

Find points of parameter field that can be inverted. We here assume that 
"""
function where_significant(X::Vector{Matrix}, tol::Real)
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
    if paraminv.case == "viscosity" || paraminv.case == "rigidity"
        println("mean $(paraminv.case): $(mean(ϕ_n))")
    elseif paraminv.case == "both"
        m = paraminv.nparams ÷ 2
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