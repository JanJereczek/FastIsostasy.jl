module FastIsostasy

using AbstractFFTs: AbstractFFTs
using CUDA: CuArray, CuMatrix, CUFFT, allowscalar
using DelimitedFiles: readdlm
using Downloads: download
using FastGaussQuadrature: gausslegendre
using FFTW: fft, ifft, plan_fft!, plan_ifft!, cFFTWPlan
using LinearAlgebra: Diagonal, det, diagm, norm
using NetCDF
using OrdinaryDiffEqTsit5: init, ODEProblem, solve, DiscreteCallback

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
@reexport using OrdinaryDiffEqTsit5: step!, Tsit5
@reexport using OrdinaryDiffEqLowOrderRK: Euler, SplitEuler, Heun, Ralston,
    Midpoint, RK4, BS3, OwrenZen3, OwrenZen4, OwrenZen5, BS5, DP5, Anas5,
    RKO65, FRK65, RKM, MSRK5, MSRK6, PSRK4p7q6, PSRK3p5q4, PSRK3p6q5, Stepanov5,
    SIR54, Alshina2, Alshina3, Alshina6

include("convenience_types.jl")
include("domain.jl")
include("boundary_conditions.jl")
include("constants.jl")
include("layering.jl")
include("convolution.jl")
include("interpolations.jl")
include("tools.jl")
include("adaptive_ocean.jl")
include("state.jl")
include("io.jl")
include("solve.jl")
include("utils.jl")
include("derivatives.jl")
include("derivatives_parallel.jl")
include("sealevel.jl")
include("material.jl")
include("mechanics.jl")
include("analytic_solutions.jl")
include("dataloaders.jl")
include("elra.jl")
include("inversion.jl")
include("coordinates.jl")

# convenience_types.jl
export EarthModel, AbstractLithosphere, AbstractMantle, AbstractRheology,
       LaterallyConstantLithosphere, LaterallyVariableLithosphere,
       LaterallyConstantMantle, LaterallyVariableMantle, RelaxedRheology,
       ViscousRheology

# domain.jl
export ComputationDomain

# boundary_conditions.jl
export CornerBC, BorderBC, DistanceWeightedBC, ProblemBCs, RegularBCSpace,
       ExtendedBCSpace

# constants.jl
export PhysicalConstants, ReferenceEarthModel

# layering.jl
export AbstractLayering, UniformLayering, ParallelLayering, EqualizedLayering
export FoldedLayering, get_layer_boundaries, interpolate2layers
export LayeredEarth

# convolution.jl
export InplaceConvolution, convolution!, blur, samesize_conv, samesize_conv!

# tools.jl
export FastIsoTools

# adaptive_ocean.jl
export OceanSurfaceChange

# state.jl
export CurrentState, ReferenceState

# io.jl
export NetcdfOutput, write_nc!, write_out!

# solve.jl
export DiffEqOptions, SolverOptions, FastIsoProblem, solve!, init_integrator, step!

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export lon360tolon180
export reinit_structs_cpu, meshgrid, kernelcollect

export get_quad_coeffs, get_r, gauss_distr, generate_gaussian_field
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export remake!, null, not, cudainfo, kernelpromote, kernelnull

# derivatives.jl
export update_second_derivatives!

# sealevel.jl
export update_loadcolumns!, update_elasticresponse!, update_dz_ss!, get_dz_ssgreen
export columnanom_load!, columnanom_full!, columnanom_ice, columnanom_water
export columnanom_litho!, columnanom_mantle!, update_z_ss!, total_volume
export update_V_af!, update_V_den!, update_V_pov!, height_above_floatation

# material.jl
export maxwelltime_scaling!, get_shearmodulus, get_rigidity, load_prem

# mechanics.jl
export elra!, lv_elva!, update_diagnostics!
export update_deformation_rhs!, build_greenintegrand, get_elasticgreen
export thinplate_horizontal_displacement

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

export apply_bc!, update_bedrock!, update_loadcolumns!, columnanom_load, update_elastic_response!, columnanom_litho, update_dz_ss!, update_maskocean!, update_bsl!, update_z_ss!, update_maskgrounded!
end