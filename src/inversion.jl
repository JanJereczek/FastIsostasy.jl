"""
    ParameterReduction

Abstract type for parameter reduction methods. Any subtype must implement the
`reconstruct!(sim, theta)` method, which assigns the reconstructed parameter
values to `sim::Simulation`.
"""
abstract type ParameterReduction{T} end

"""
    InversionConfig

Struct containing configuration parameters for an [`InversionProblem`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct InversionConfig{T<:AbstractFloat}
    "inversion method to use"
    method::Any
    "number of ensemble members"
    N_ens::Int
    "number of iterations for the inversion"
    N_iter::Int
    "number of noisy samples drawn from the observations"
    n_samples::Int
    """
    regularization factor. With enough observational data, `α = 1` (no
    regularization).
    """
    α_reg::T
    """
    update frequency for the inversion. `1`: approximate the posterior covariance
    matrix with an uninformative prior. `0`: weighted average between the posterior
    covariance matrix with an uninformative prior, and the prior.
    """
    update_freq::Int
    "scaling factor for the observational covariance matrix"
    scale_obscov::T
end

function InversionConfig(
    method,
    N_ens,
    N_iter,
    n_samples;
    α_reg = 1.0,
    update_freq = 1,
    scale_obscov = 1_000.0,
)
    return InversionConfig(
        method,
        N_ens,
        N_iter,
        n_samples,
        α_reg,
        update_freq,
        scale_obscov,
    )
end

"""
    InversionData

Struct containing the inversion data.

# Fields
$(TYPEDFIELDS)
"""
struct InversionData{T<:AbstractFloat,M<:Matrix{T}}
    "time vector"
    t::Vector{T}
    "ground truth response"
    Y::Vector{M}
    "number of output time steps used for inversion"
    nY::Int
    "region of interest"
    mask::BitMatrix
    "`count(mask)`, the number of cells used for inversion"
    countmask::Int
end

function InversionData(t, Y, mask)
    nY = length(Y)
    countmask = count(mask)
    return InversionData(t, Y, nY, mask, countmask)
end

"""
    InversionProblem

Struct containing variables and configs for the inversion of
Solid-Earth parameter fields. `InversionProblem` needs to be initialized
using [`inversion_problem`](@ref). For now, the unscented Kalman inversion
is the only method available.

# Fields
$(TYPEDFIELDS)
"""
struct InversionProblem{
    T<:AbstractFloat,
    V<:Vector{T},
    M<:Matrix{T},
    R<:ParameterReduction{T},
    PD,
    EKP,
}
    "the [`Simulation`](@ref) template the forward runs are made from"
    sim::Simulation{T,<:Any,M,<:Any,<:Any,<:Any,<:Any,<:Any}
    "the [`InversionConfig`](@ref) for the inversion"
    config::InversionConfig# {T}
    "the [`InversionData`](@ref) for the inversion"
    data::InversionData{T,M}
    "the [`ParameterReduction`](@ref) method"
    reduction::R
    "the prior distribution"
    priors::PD
    "the unscented Kalman inversion object"
    ukiobj::EKP
    "the error at each iteration"
    error::V
    "the mean parameter vector at each saved iteration"
    out::Vector{V}
    "the ensemble of forward responses"
    G_ens::M
end


"""
    inversion_problem(sim, config, data, reduction, priors; save_stride_iter::Int = 1)

Generate an inversion problem for the given `sim::Simulation` object.
"""
function inversion_problem end

function run! end
function forward_fastiso end

"""
    print_inversion_evolution(paraminv, n, ϕ_n, reduction)

Print the inversion evolution.
"""
function print_inversion_evolution end

"""
    extract_inversion(paraminv, n)

Extract the inversion results to compare them with the ground truth.
"""
function extract_inversion end

"""
    reconstruct!(sim, params, reduction)

Reconstruct the parameter values from `reduction` and update `sim` accordingly.
"""
function reconstruct! end

"""
    extract_output(sim, reduction, data)

Extract the output of the forward run for the inversion.
"""
function extract_output end

export inversion_problem,
    run!,
    forward_fastiso,
    print_inversion_evolution,
    extract_inversion,
    reconstruct!,
    extract_output
