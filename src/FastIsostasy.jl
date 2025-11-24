module FastIsostasy

using AbstractFFTs: AbstractFFTs
using CUDA: CuArray, CuMatrix, CUFFT, allowscalar
using DelimitedFiles: readdlm
using DocStringExtensions
using Downloads: download
using FiniteDifferences: central_fdm, forward_fdm, backward_fdm
using FastGaussQuadrature: gausslegendre
using FFTW: fft, ifft, plan_fft!, plan_ifft!, cFFTWPlan, rFFTWPlan, plan_rfft, plan_irfft
using LinearAlgebra: Diagonal, det, diagm, norm, mul!
using NetCDF
using OrdinaryDiffEqTsit5: init, ODEProblem, solve, DiscreteCallback, CallbackSet

using ParallelStencil: ParallelStencil, @init_parallel_stencil, @parallel, @parallel_indices
using Statistics: mean, cov, std
using SpecialFunctions: besselj0, besselj1, besselk

# Init stencil on GPU. Will only be used if specified in RegionalDomain.
allowscalar(false)
@init_parallel_stencil(CUDA, Float32, 3);

using Reexport: Reexport, @reexport
@reexport using Interpolations
@reexport using Proj
@reexport using OrdinaryDiffEqTsit5: step!, Tsit5
@reexport using OrdinaryDiffEqLowOrderRK: Euler, SplitEuler, Heun, Ralston,
    Midpoint, RK4, BS3, OwrenZen3, OwrenZen4, OwrenZen5, BS5, DP5, Anas5,
    RKO65, FRK65, RKM, MSRK5, MSRK6, PSRK4p7q6, PSRK3p5q4, PSRK3p6q5, Stepanov5,
    SIR54, Alshina2, Alshina3, Alshina6

include("convenience_types.jl")
include("interpolations.jl")
include("barystatic_sealevel.jl")
include("domain.jl")
include("boundary_conditions.jl")
include("constants.jl")
include("layering.jl")
include("material.jl")
include("solidearth.jl")
include("convolutions.jl")
include("tools.jl")
include("state.jl")
include("io.jl")
include("simulation.jl")
include("loads.jl")
include("topography.jl")
include("utils.jl")
include("derivatives.jl")
include("derivatives_parallel.jl")
include("sealevel.jl")
include("deformation.jl")
include("analytic_solutions.jl")
include("dataloaders.jl")
include("inversion.jl")
include("coordinates.jl")

# interpolations.jl
# export TimeInterpolation0D, TimeInterpolation2D, interpolate!

# barystatic_sealevel.jl
export AbstractUpdateBSL, InternalUpdateBSL, ExternalUpdateBSL, ReferenceBSL
export AbstractBSL, ConstantBSL, ConstantOceanSurfaceBSL, PiecewiseConstantBSL
export PiecewiseLinearBSL, ImposedBSL, CombinedBSL
export update_bsl!

# domain.jl
export AbstractDomain, RegionalDomain, GlobalDomain

# boundary_conditions.jl
export BoundaryConditions
export AbstractIceThickness, TimeInterpolatedIceThickness, ExternallyUpdatedIceThickness
export AbstractBCSpace, RegularBCSpace, ExtendedBCSpace
export AbstractBC, OffsetBC, NoBC
export CornerBC, BorderBC, DistanceWeightedBC, MeanBC
export apply_bc!

# constants.jl
export PhysicalConstants    #, ReferenceSolidEarthModel

# layering.jl
export AbstractLayering
export UniformLayering, ParallelLayering, EqualizedLayering, FoldedLayering
export get_layer_boundaries, interpolate2layers

# convolutions.jl
# export ConvolutionPlan, convo!, nextfastfft, _zeropad!, samesize_conv!
export gaussian_smooth, conv!, ConvolutionPlan, ConvolutionPlanHelpers

# tools.jl
export GIATools

# state.jl
export CurrentState, ReferenceState

# io.jl
export NetcdfOutput, NativeOutput, write_nc!, write_out!
export PaddedOutputCrop, AsymetricOutputCrop

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export lon360tolon180
export reinit_structs_cpu, meshgrid, kernelcollect

export get_quad_coeffs, get_r, gauss_distr, generate_gaussian_field
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export zeros, not, cudainfo, kernelpromote, kernelzeros

# derivatives.jl
export update_second_derivatives!   #, dxx!, dyy!, FiniteDiffParams

# loads.jl
export height_above_floatation, columnanom_water!

# topography.jl
# export update_Haf!, update_bedrock!
# export update_maskocean!, update_maskgrounded!
export get_maskgrounded, get_maskocean

# sealevel.jl
export RegionalSeaLevel
export AbstractSeaSurface, AbstractSealevelLoad
export NoSealevelLoad, InteractiveSealevelLoad
export LaterallyConstantSeaSurface, LaterallyVariableSeaSurface
export AbstractVolumeContribution, GoelzerVolumeContribution, NoVolumeContribution
export AbstractAdjustmentContribution, GoelzerAdjustmentContribution, NoAdjustmentContribution
export AbstractDensityContribution, GoelzerDensityContribution, NoDensityContribution
export update_dz_ss!

# material.jl
export AbstractCalibration, NoCalibration, SeakonCalibration, apply_calibration!

export AbstractCompressibility, IncompressibleMantle, CompressibleMantle, apply_compressibility!

export AbstractViscosityLumping, TimeDomainViscosityLumping
export FreqDomainViscosityLumping, MeanViscosityLumping, MeanLogViscosityLumping
export get_effective_viscosity_and_scaling, green_viscous

export get_relaxation_time, get_relaxation_time_weaker, get_relaxation_time_stronger
export get_rigidity, get_shearmodulus, get_elastic_green, get_flexural_lengthscale
export calc_viscous_green

# solidearth.jl
export SolidEarth
export AbstractLithosphere, AbstractMantle
export RigidLithosphere, LaterallyConstantLithosphere, LaterallyVariableLithosphere
export RigidMantle, RelaxedMantle, MaxwellMantle
export AbstractLithosphereColumn, IncompressibleLithosphereColumn, CompressibleLithosphereColumn

# deformation.jl
export update_dudt!, update_deformation_rhs!, thinplate_horizontal_displacement
export update_elasticresponse!

# analytic solutions
export analytic_solution

# data loaders
export load_dataset, get_greenintegrand_coeffs
export load_etopo2022, load_wiens2022
export load_lithothickness_pan2022, load_logvisc_pan2022
export load_ice6gd
export load_spada2011, spada_cases
export load_latychev_test3, load_latychev2023_ICE6G

# simulation.jl
export DiffEqOptions, SolverOptions, Simulation, run!, init_integrator, step!
export update_diagnostics!

include("plots.jl")

# inversion.jl
export InversionConfig, InversionData, InversionProblem, ParameterReduction
export ViscositySnippet

end