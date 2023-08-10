module FastIsostasy

using LinearAlgebra
using Statistics: mean
using Distributions: MvNormal
using JLD2: jldopen
using DelimitedFiles: readdlm
using Interpolations: linear_interpolation, Flat
using FFTW: plan_fft, plan_ifft
using AbstractFFTs
using FastGaussQuadrature: gausslegendre
using DSP: conv
using CUDA: CuArray, CUFFT, allowscalar
using OrdinaryDiffEq: ODEProblem, solve, remake, OrdinaryDiffEqAlgorithm
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
using SpecialFunctions: besselj0, besselj1
using DynamicalSystemsBase: CoupledODEs, trajectory

using Reexport
@reexport using Interpolations
@reexport using OrdinaryDiffEq: Euler, Midpoint, Heun, Ralston, BS3, BS5, RK4,
    OwrenZen3, OwrenZen4, OwrenZen5, Tsit5, DP5, RKO65, TanYam7, DP8,
    Feagin10, Feagin12, Feagin14, TsitPap8, Vern6, Vern7, Vern8, Vern9,
    SSPRK22, SSPRK33, SSPRK53, SSPRK63, SSPRK73, SSPRK83, SSPRK432, SSPRK43,
    SSPRK932, SSPRK54, SSPRK104, SSPRKMSVS32, SSPRKMSVS43, SSPRK53_2N1, SSPRK53_2N2

include("structs.jl")
include("utils.jl")
include("derivatives.jl")
include("integrators.jl")
include("geostate.jl")
include("mechanics.jl")
include("inversion.jl")
include("analytic_solutions.jl")

# structs.jl
export ComputationDomain, PhysicalConstants
export ReferenceEarthModel, LateralVariability
export GeoState, RefGeoState
export FastIsoTools, FastIsoProblem

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export meshgrid, dist2angulardist, latlon2stereo, stereo2latlon
export matrify, kernelpromote, reinit_structs_cpu

export loginterp_viscosity, get_rigidity, load_prem
export maxwelltime_scaling!, compute_shearmodulus

export get_r, gauss_distr, samesize_conv
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export write_out!

export quadrature1D
export meshgrid
export get_quad_coeffs
export get_elasticgreen

# derivatives.jl
export mixed_fdx, mixed_fdy, mixed_fdxx, mixed_fdyy
export get_differential_fourier
export central_fdx, mixed_fdx, mixed_fdx!
export dxx!, dyy!, dxy!, dx!, dy!

# mechanics.jl
export dudt_isostasy!, update_diagnostics!
export simple_euler!, SimpleEuler
export init, solve!, step!
export corner_bc!, no_bc
export update_loadcolumns!, update_elasticresponse!, update_geoid!, update_sealevel!

# geostate.jl
export columnanom_load, correct_surfacedisctortion, columnanom_full

# inversion.jl
export ParamInversion, perform, extract_inversion

# analytic solutions
export analytic_solution

end
