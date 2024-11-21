module FastIsostasy

using AbstractFFTs: AbstractFFTs
using CUDA: CuArray, CuMatrix, CUFFT, allowscalar
using DelimitedFiles: readdlm
using Downloads: download
using FastGaussQuadrature: gausslegendre
using FFTW: fft, ifft, plan_fft!, plan_ifft!, cFFTWPlan
using LinearAlgebra: Diagonal, det, diagm, norm
using NetCDF
using NLsolve: mcpsolve
using OrdinaryDiffEqTsit5: init, Tsit5, ODEProblem, solve, DiscreteCallback
using ParallelStencil: ParallelStencil, @init_parallel_stencil, @parallel,
                       @parallel_indices
using Statistics: mean, cov, std
using SpecialFunctions: besselj0, besselj1

# Init stencil on GPU. Will only be used if specified in ComputationDomain.
allowscalar(false)
@init_parallel_stencil(CUDA, Float64, 3);

using Reexport: Reexport, @reexport
@reexport using Interpolations
@reexport using Proj
@reexport using OrdinaryDiffEqTsit5: step!

include("convenience_types.jl")
include("convolution.jl")
include("adaptive_ocean.jl")
include("layering.jl")
include("structs.jl")
include("utils.jl")
include("derivatives.jl")
include("derivatives_parallel.jl")
include("geostate.jl")
include("material.jl")
include("mechanics.jl")
include("analytic_solutions.jl")
include("dataloaders.jl")
include("elra.jl")
include("inversion.jl")
include("coordinates.jl")

# convolution.jl
export InplaceConvolution

# structs.jl
export ComputationDomain, PhysicalConstants
export ReferenceEarthModel, LayeredEarth, SolverOptions
export CurrentState, ReferenceState, NetcdfOutput
export FastIsoTools, FastIsoProblem

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export lon360tolon180
export reinit_structs_cpu, meshgrid, kernelcollect

export get_quad_coeffs, get_r, gauss_distr, generate_gaussian_field, samesize_conv, blur
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export write_nc!, write_out!, remake!, null, not, cudainfo

# adaptive_ocean.jl
export OceanSurfaceChange

# geostate.jl
export update_loadcolumns!, update_elasticresponse!, update_dz_ss!, get_dz_ssgreen
export columnanom_load!, columnanom_full!, columnanom_ice, columnanom_water
export columnanom_litho!, columnanom_mantle!, update_z_ss!, total_volume
export update_V_af!, update_V_den!, update_V_pov!, height_above_floatation

# layering.jl
export AbstractLayering, UniformLayering, ParallelLayering, EqualizedLayering
export FoldedLayering, get_layer_boundaries

# material.jl
export maxwelltime_scaling!, get_shearmodulus, get_rigidity, load_prem

# mechanics.jl
export init_integrator, solve!, elra!, lv_elva!, update_diagnostics!  # step!
export update_deformation_rhs!, build_greenintegrand, get_elasticgreen
export get_horizontal_displacement

# elra.jl
export get_flexural_lengthscale, get_kei, calc_kei_value, calc_viscous_green

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
export InversionConfig, InversionData, InversionProblem, ParameterReduction,
    ViscositySnippet, extract_viscous_displacement, extract_elastic_displacement,
    extract_total_displacement

export update_second_derivatives!

end