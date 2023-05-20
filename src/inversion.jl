"""
    ParamInversion

Return a struct containing all variables and configs related to the inversion of
Solid-Earth parameter fields based on displacement fields. For now, only viscosity
can be inverted but future versions will support lithosphere rigidity.
For now, the unscented Kalman inversion is the only method available but ensemble
Kalman inversion will be available in future.
"""
# For now, method is fixed to UKI
struct ParamInversion
    Omega::ComputationDomain
    c::PhysicalConstants
    p::MultilayerEarth
    t::Vector
    y::Vector
    Hice::Vector{Matrix}
    obs_idx::Matrix
    case::String
    method::Any
    nt::Int
    nparams::Int
    nobs::Int
    paramspriors::NamedTuple
    N_iter::Int
    α_reg::Real
    update_freq::Int
    n_samples::Int
    μ_y::Vector
    Σ_y::Matrix
end

function ParamInversion(
    t::Vector,
    u::Vector{Matrix},
    Hice::Vector{Matrix};
    L = 3000e3,
    kwargs...,
)
    n1, n2 = size(u[1])
    Omega = ComputationDomain(L, n1)
    c = PhysicalConstants()
    p = MultilayerEarth(Omega, c)
    return ParamInversion(Omega, c, p, t, u, Hice; kwargs...)
end

function ParamInversion(
    Omega::ComputationDomain,
    c::PhysicalConstants,
    p::MultilayerEarth,
    t::Vector,
    U::Vector{Matrix{T}},
    Hice::Vector{Matrix{T}};
    case::String = "viscosity",
    method::Any = Unscented,
    Htol::Real = 1.0,
    paramspriors::NamedTuple = defaultpriors(case),
    N_iter::Int = 20,
    α_reg::Real = 1.0,
    update_freq::Int = 1,
    n_samples::Int = 100,
    scale_obscov::Real = 10000.0,
) where {T<:AbstractFloat}
    if (length(t) != length(U)) ||
        (length(t) != length(Hice)) ||
        (length(U) != length(Hice))
        error("The length of the provided a time vector, displacement field history and load history do not match!")
    end

    maxHtransient = max.( [abs.(H) for H in Hice]... )
    obs_idx = where_response(maxHtransient, Htol)
    significantload = vcat( [H[obs_idx] for H in Hice]... )
    nt = length(t)
    nparams = sum(obs_idx)
    nobs = nt * nparams

    if nobs != length(significantload)
        error("The number of observations with significant loading do not correspond to the dimension of the Kalman problem.")
    end

    y = vcat([u[obs_idx] for u in U]...)

    # Generating noisy observations
    μ_y = zeros(nobs)
    Σ_y = uncorrelated_obs_covariance(scale_obscov, significantload)

    return ParamInversion(Omega, c, p, t, y, Hice, obs_idx, case, method, nt, nparams, nobs,
        paramspriors, N_iter, α_reg, update_freq, n_samples, μ_y, Σ_y)

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

    ynoisy = zeros(paraminv.nobs, paraminv.n_samples)
    for i in 1:paraminv.n_samples
        ynoisy[:, i] = paraminv.y .+ rand(MvNormal(paraminv.μ_y, paraminv.Σ_y))
    end
    truth = Observations.Observation(ynoisy, paraminv.Σ_y, ["Noisy truth"])
    truth_sample = truth.mean

    priors = combine_distributions([constrained_gaussian( "p_$(i)",
        paraminv.paramspriors.mean, paraminv.paramspriors.var,
        paraminv.paramspriors.lowerbound, paraminv.paramspriors.upperbound) for i in 1:paraminv.nparams])

    # Here we also could use process = Inversion()
    process = Unscented(mean(priors), cov(priors);
        α_reg = paraminv.α_reg, update_freq = paraminv.update_freq)
    ukiobj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)
    err = zeros(paraminv.N_iter)
    for n in 1:paraminv.N_iter
        ϕ_n = get_ϕ_final(priors, ukiobj)       # Params in physical/constrained space
        G_n = [fastisostasy(ϕ_n[:, j], paraminv) for j in axes(ϕ_n, 2)]      # Evaluate forward map
        G_ens = hcat(G_n...)
        EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
        err[n] = get_error(ukiobj)[end]
        print_kalmanprocess_evolution(paraminv, n, ϕ_n, err[n], ukiobj.process.uu_cov[n])
    end
    return priors, ukiobj
end

# ϕ_n = get_ϕ_final(priors, ukiobj)

function fastisostasy(params::Vector, paraminv::ParamInversion)
    if paraminv.case == "viscosity"
        paraminv.p.effective_viscosity[paraminv.obs_idx] .= 10 .^ params
    elseif paraminv.case == "rigidity"
        paraminv.p.lithosphere_rigidity[paraminv.obs_idx] .= params
    elseif paraminv.case == "both"
        paraminv.p.effective_viscosity[paraminv.obs_idx] .= 10 .^ params[1:mparams]
        paraminv.p.lithosphere_rigidity[paraminv.obs_idx] .= params[mparams+1:end]
    end

    t = vcat(0.0, paraminv.t)
    results = fastisostasy( t, paraminv.Omega, paraminv.c,
        paraminv.p, paraminv.Hice[1], ODEsolver=BS3(), verbose = false)
    Gx = vcat([reshape(results.viscous[j][paraminv.obs_idx],
        paraminv.nparams) for j in eachindex(results.viscous)[2:end]]...)
    # results are only taken from the 2nd index onwards because
    # the first index returns the solution at time t=0.
    return Gx
end

"""
    where_response()

Find points of parameter field that can be inverted. We here assume that 
"""
function where_response(load, loadtol)
    return abs.(load) .> loadtol
end

function uncorrelated_obs_covariance(scale_obscov, significantload)
    diagvar = scale_obscov ./ (significantload .+ 1)   # 10000.0
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

function extract_inversion(priors, ukiobj, paraminv)
    p = get_ϕ_mean_final(priors, ukiobj)
    Gx = get_g_mean_final(ukiobj)
    e_mean = mean(abs.(paraminv.y - Gx))
    e_sort = sort(abs.(paraminv.y - Gx))
    return p, Gx, e_mean, e_sort
end