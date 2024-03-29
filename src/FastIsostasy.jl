module FastIsostasy

using LinearAlgebra
using Statistics: mean, cov
using Distributions: MvNormal
using .Threads
using Downloads

using JLD2
using NCDatasets
using DelimitedFiles: readdlm

using Interpolations: linear_interpolation, Flat
using NLsolve: mcpsolve
using FFTW: fft, ifft, plan_fft!, plan_ifft!, cFFTWPlan
using AbstractFFTs
using FastGaussQuadrature: gausslegendre
using CUDA: CuArray, CuMatrix, CUFFT, allowscalar
using OrdinaryDiffEq: ODEProblem, solve, remake, OrdinaryDiffEqAlgorithm
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
using SpecialFunctions: besselj0, besselj1
using DynamicalSystemsBase: CoupledODEs, trajectory
using ParallelStencil

using Reexport
@reexport using Interpolations
@reexport using OrdinaryDiffEq: Euler, Midpoint, Heun, Ralston, BS3, BS5, RK4,
    OwrenZen3, OwrenZen4, OwrenZen5, Tsit5, DP5, RKO65, TanYam7, DP8,
    Feagin10, Feagin12, Feagin14, TsitPap8, Vern6, Vern7, Vern8, Vern9,
    SSPRK22, SSPRK33, SSPRK53, SSPRK63, SSPRK73, SSPRK83, SSPRK432, SSPRK43,
    SSPRK932, SSPRK54, SSPRK104, SSPRKMSVS32, SSPRKMSVS43, SSPRK53_2N1, SSPRK53_2N2,
    KenCarp47

include("convenience_types.jl")
include("convolution.jl")
include("adaptive_ocean.jl")
include("structs.jl")
include("utils.jl")
include("derivatives.jl")
include("derivatives_parallel.jl")
include("integrators.jl")
include("geostate.jl")
include("mechanics.jl")
include("inversion.jl")
include("analytic_solutions.jl")
include("dataloaders.jl")

# convolution.jl
export InplaceConvolution

# structs.jl
export ComputationDomain, PhysicalConstants
export ReferenceEarthModel, LayeredEarth, SolverOptions
export CurrentState, ReferenceState
export FastIsoTools, FastIsoProblem

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export latlon2stereo, stereo2latlon, lon360tolon180
export reinit_structs_cpu, meshgrid, kernelcollect

export get_r, gauss_distr, generate_gaussian_field, samesize_conv, blur
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export write_out!, remake!, savefip, null, not

# adaptive_ocean.jl
export OceanSurfaceChange

# geostate.jl
export update_loadcolumns!, update_elasticresponse!, update_geoid!
export columnanom_load!, columnanom_full!, columnanom_ice, columnanom_water
export columnanom_litho!, columnanom_mantle!, update_seasurfaceheight!, total_volume
export update_V_af!, update_V_den!, update_V_pov!, height_above_floatation

# mechanics.jl
export init, solve!, step!, dudt_isostasy!, update_diagnostics!
export maxwelltime_scaling!, compute_shearmodulus, get_rigidity, load_prem

# integrators.jl
export simple_euler!, SimpleEuler

# inversion.jl
export InversionConfig, InversionData, InversionProblem, solve, extract_inversion
export get_ϕ_mean_final     # from EnsembleKalmanProcesses.jl

# analytic solutions
export analytic_solution

# data loaders
export load_dataset, get_greenintegrand_coeffs
export load_etopo2022, bathymetry, load_wiens2022
export load_lithothickness_pan2022, load_logvisc_pan2022
export load_ice6gd
export load_spada2011, spada_cases
export load_latychev_test3, load_latychev2023_ICE6G

end