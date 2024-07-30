"""
    ParameterReduction

Abstract type for parameter reduction methods. Any subtype must implement the
`reconstruct!(fip, theta)` method, which assigns the reconstructed parameter
values to `fip::FastIsoProblem`.
"""
abstract type ParameterReduction end


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
    scale_obscov = 10_000.0,
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
Solid-Earth parameter fields. For now, only viscosity can be inverted but future
versions will support lithosphere rigidity. For now, the unscented Kalman inversion
is the only method available but ensemble Kalman inversion will be available in future.
"""
struct InversionProblem{T<:AbstractFloat, M<:Matrix{T}, R<:ParameterReduction}
    fip::FastIsoProblem{T, <:Any, M, <:Any, <:Any, <:Any, <:Any}
    config::InversionConfig
    data::InversionData{T, M}
    reduction::R
    priors::Any
    ukiobj::Any
    error::Vector{T}
    out::Vector{Vector{T}}
    G_ens::M
end
# priors::ParameterDistribution
# ukiobj::EnsembleKalmanProcess   # {T, Int64, Unscented{T, Int64}, DefaultScheduler{T}}

"""
    where_significant()

Find points of parameter field that can be inverted. We here assume that 
"""
function where_significant(X::Vector{<:Matrix}, tol::Real)
    transientmax = max.( [abs.(x) for x in X]... )
    return transientmax .> tol
end

function testfunc end
function inversion_problem end
function solve! end
function forward_fastiso end
function print_inversion_evolution end
function extract_inversion end
function reconstruct! end
function extract_output end

export testfunc, inversion_problem, solve!, forward_fastiso,
    print_inversion_evolution, extract_inversion, reconstruct!, extract_output