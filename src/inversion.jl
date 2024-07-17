"""
    InversionConfig

Struct containing configuration parameters for a [`InversionProblem`].

Need to choose regularization factor α ∈ (0,1],  
When you have enough observation data α=1: no regularization

update_freq 1 : approximate posterior cov matrix with an uninformative prior
            0 : weighted average between posterior cov matrix with an uninformative prior and prior
"""
struct InversionConfig
    case::String
    method::Any
    paramspriors::NamedTuple
    N_iter::Int
    α_reg::Real
    update_freq::Int
    n_samples::Int
    scale_obscov::Real
end

function InversionConfig(;
    case = "viscosity",
    method = nothing,
    paramspriors = defaultpriors(case),
    N_iter = 20,
    α_reg = 1.0,
    update_freq = 1,
    n_samples = 100,
    scale_obscov = 10_000.0,
)
    if method == nothing
        throw(ArgumentError("Please provide a method for the inversion."))
    end
    if case ∉ ["viscosity", "rigidity", "both"]
        throw(ArgumentError("Please provide a valid case for inversion."))
    end
    return InversionConfig(case, method, paramspriors, N_iter, α_reg, update_freq,
        n_samples, scale_obscov)
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
    priors::Any
    ukiobj::Any
    # priors::ParameterDistribution
    # ukiobj::EnsembleKalmanProcess   # {T, Int64, Unscented{T, Int64}, DefaultScheduler{T}}
    error::Vector{T}
    out::Vector{Vector{T}}
    G_ens::M
end


"""
    where_significant()

Find points of parameter field that can be inverted. We here assume that 
"""
function where_significant(X::Vector{<:Matrix}, tol::Real)
    transientmax = max.( [abs.(x) for x in X]... )
    return transientmax .> tol
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

function testfunc end
function inversion_problem end
function solve! end
function forward_fastiso end
function print_inversion_evolution end
function extract_inversion end

export testfunc, inversion_problem, solve!, forward_fastiso,
    print_inversion_evolution, extract_inversion