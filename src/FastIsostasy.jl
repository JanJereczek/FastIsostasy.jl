module FastIsostasy

using AbstractFFTs: AbstractFFTs
using CUDA: CuArray, CuMatrix, CUFFT, allowscalar
using DelimitedFiles: readdlm
using Downloads: download
using DynamicalSystemsBase: CoupledODEs, trajectory
using FastGaussQuadrature: gausslegendre
using FFTW: fft, ifft, plan_fft!, plan_ifft!, cFFTWPlan
using Interpolations: Flat, Gridded, Linear, OnGrid, Throw, linear_interpolation
using JLD2: @load, jldopen, load
using LinearAlgebra: Diagonal, det, diagm, norm
using NCDatasets: NCDatasets, NCDataset, defDim, defVar
using NLsolve: mcpsolve
using OrdinaryDiffEq: ODEProblem, solve, remake, OrdinaryDiffEqAlgorithm
using ParallelStencil: ParallelStencil, @init_parallel_stencil, @parallel,
                       @parallel_indices
using Statistics: mean, cov
using SpecialFunctions: besselj0, besselj1
# using .Threads

using Reexport: Reexport, @reexport
@reexport using Interpolations
@reexport using OrdinaryDiffEq: Euler, Midpoint, Heun, Ralston, BS3, BS5, RK4,
    OwrenZen3, OwrenZen4, OwrenZen5, Tsit5, DP5, RKO65, TanYam7, DP8,
    Feagin10, Feagin12, Feagin14, TsitPap8, Vern6, Vern7, Vern8, Vern9

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
include("analytic_solutions.jl")
include("dataloaders.jl")
include("elra.jl")
include("inversion.jl")

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

export get_r, gauss_distr, generate_gaussian_field, samesize_conv, samesize_conv!, blur
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export write_out!, remake!, savefip, null, not, cudainfo

# adaptive_ocean.jl
export OceanSurfaceChange

# geostate.jl
export update_loadcolumns!, update_elasticresponse!, update_geoid!
export columnanom_load!, columnanom_full!, columnanom_ice, columnanom_water
export columnanom_litho!, columnanom_mantle!, update_seasurfaceheight!, total_volume
export update_V_af!, update_V_den!, update_V_pov!, height_above_floatation

# mechanics.jl
export init, solve!, step!, lv_elva!, update_diagnostics!
export maxwelltime_scaling!, compute_shearmodulus, get_rigidity, load_prem
export get_flexural_lengthscale, get_kei, calc_kei_value, calc_viscous_green
export update_deformation_rhs!
export build_greenintegrand, get_quad_coeffs, get_elasticgreen, get_geoidgreen

# integrators.jl
export simple_euler!, SimpleEuler

# analytic solutions
export analytic_solution

# data loaders
export load_dataset, get_greenintegrand_coeffs
export load_etopo2022, load_wiens2022
export load_lithothickness_pan2022, load_logvisc_pan2022
export load_ice6gd
export load_spada2011, spada_cases
export load_latychev_test3, load_latychev2023_ICE6G

# inversion.jl
export InversionConfig, InversionData, InversionProblem

export update_second_derivatives!

end