"""
    ParameterReduction

Abstract type for parameter reduction methods. Any subtype must implement the
`reconstruct!(fip, theta)` method, which assigns the reconstructed parameter
values to `fip::FastIsoProblem`.
"""
abstract type ParameterReduction{T} end


"""
    extract_viscous_displacement(fip::FastIsoProblem{T, L, M})

Extract the viscous displacement field from the `fip::FastIsoProblem` object.
"""
extract_viscous_displacement(fip) = fip.out.u

"""
    extract_elastic_displacement(fip::FastIsoProblem{T, L, M})

Extract the elastic displacement field from the `fip::FastIsoProblem` object.
"""
extract_elastic_displacement(fip) = fip.out.ue

"""
    extract_total_displacement(fip::FastIsoProblem{T, L, M})

Extract the total displacement field from the `fip::FastIsoProblem` object.
"""
extract_total_displacement(fip) = fip.out.u .+ fip.out.ue

"""
    InversionConfig

Struct containing configuration parameters for a [`InversionProblem`].

# Fields

- `method::Any`: Inversion method to use.
- `paramspriors::NamedTuple`: Prior information about the parameters to invert.
- `N_iter::Int`: Number of iterations for the inversion.
- `α_reg::Real`: Regularization factor. When you have enough observation data α=1 (no regularization)
- `update_freq::Int`: Update frequency for the inversion.
1 : approximate posterior cov matrix with an uninformative prior.
0 : weighted average between posterior cov matrix with an uninformative prior and prior.
- `n_samples::Int`: Number of samples for the inversion.
- `scale_obscov::Real`: Scaling factor for the observational covariance matrix.
"""
struct InversionConfig{T<:AbstractFloat}
    method::Any
    N_iter::Int
    α_reg::T
    update_freq::Int
    n_samples::Int
    scale_obscov::T
end

function InversionConfig(
    method;
    N_iter = 5,
    α_reg = 1.0,
    update_freq = 1,
    n_samples = 100,
    scale_obscov = 1_000.0,
)
    return InversionConfig(method, N_iter, α_reg, update_freq, n_samples, scale_obscov)
end

"""
    InversionData

Struct containing the inversion data.

# Fields

- `t::Vector{T}`: Time vector.
- `nt::Int`: Number of time steps.
- `X::Vector{M}`: Ground truth input (forcing).
- `Y::Vector{M}`: Ground truth response.
- `mask::BitMatrix`: Region of interest.
- `countmask::Int`: count(mask) = number of cells used for inversion.
"""
struct InversionData{T<:AbstractFloat, M<:Matrix{T}}
    t::Vector{T}        # Time vector
    nt::Int             # number of time steps
    X::Vector{M}        # Ground truth input (forcing)
    Y::Vector{M}        # Ground truth response
    mask::BitMatrix     # Region of interest
    countmask::Int      # count(mask) = number of cells used for inversion
end

"""
    InversionProblem

Struct containing variables and configs for the inversion of
Solid-Earth parameter fields. `InversionProblem` needs to be initialized
using [`inversion_problem`](@ref). For now, the unscented Kalman inversion
is the only method available.

# Fields
- `fip::FastIsoProblem`: FastIsoProblem object.
- `config::InversionConfig`: Configuration for the inversion.
- `data::InversionData`: Data for the inversion.
- `reduction::R`: Parameter reduction method.
- `priors::PD`: Prior distribution.
- `ukiobj::EKP`: Unscented Kalman inversion object.
- `error::V`: Error vector.
- `out::Vector{V}`: Output vector.
- `G_ens::M`: Ensemble of the covariance matrix.
"""
struct InversionProblem{T<:AbstractFloat, V<:Vector{T}, M<:Matrix{T},
    R<:ParameterReduction{T}, PD, EKP}
    fip::FastIsoProblem{T, <:Any, M, <:Any, <:Any, <:Any, <:Any}
    config::InversionConfig# {T}
    data::InversionData{T, M}
    reduction::R
    priors::PD
    ukiobj::EKP
    error::V
    out::Vector{V}
    G_ens::M
end

"""
    inversion_problem(fip, config, data, reduction, priors; save_stride_iter::Int = 1)

Generate an inversion problem for the given `fip::FastIsoProblem` object.
"""
function inversion_problem end

function solve! end
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
    reconstruct!(fip, params, reduction)

Reconstruct the parameter values from `reduction` and update `fip` accordingly.
"""
function reconstruct! end

"""
    extract_output(fip, reduction, data)

Extract the output of the forward run for the inversion.
"""
function extract_output end

export inversion_problem, solve!, forward_fastiso,
    print_inversion_evolution, extract_inversion, reconstruct!, extract_output