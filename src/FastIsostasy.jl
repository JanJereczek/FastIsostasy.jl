module FastIsostasy

using AbstractFFTs: AbstractFFTs
using DelimitedFiles: readdlm
using DocStringExtensions
using Downloads: download
using FastGaussQuadrature: gausslegendre
using FFTW: fft, ifft, plan_fft, plan_ifft, plan_rfft, plan_irfft, MEASURE
import LinearAlgebra
using LinearAlgebra: Diagonal, det, diagm, norm, mul!, dot
using NetCDF
using ProgressMeter: Progress, update!, finish!

using KernelAbstractions:
    KernelAbstractions, @kernel, @index, get_backend, synchronize, Backend, CPU
using Statistics: mean, cov, std
using SpecialFunctions: besselj0, besselj1, besselk

using Reexport: Reexport, @reexport
@reexport using Interpolations
@reexport using Proj

include("reductions.jl")
include("interpolations.jl")
include("barystatic_sealevel.jl")
include("domain.jl")
include("boundary_conditions.jl")
include("constants.jl")
include("transitions.jl")
include("layering.jl")
include("material.jl")
include("solidearth.jl")
include("convolutions.jl")
include("tools.jl")
include("state.jl")
include("snapshot.jl")
include("io.jl")
# Before simulation.jl: `SolverOptions` bounds its integrator field by
# `AbstractIntegrator`, which must exist when that struct is defined. Everything
# integrators.jl needs from later files (`Simulation`, `update_diagnostics!`,
# `spectral_radius_estimate`) is referenced from function bodies only.
include("integrators.jl")
include("progress.jl")
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
include("stability.jl")

# inverse problem API (new; src/inverse/)
include("inverse/diffmode.jl")
include("inverse/observables.jl")
include("inverse/encodings.jl")
include("inverse/regularization.jl")
include("inverse/problem.jl")
include("inverse/recording.jl")

# interpolations.jl
export TimeInterpolation0D, TimeInterpolation2D, interpolate!

# barystatic_sealevel.jl
export AbstractBSLUpdate, InternalBSLUpdate, ExternalBSLUpdate, ReferenceBSL
export AbstractBSL, ConstantBSL, ConstantOceanSurfaceBSL, PiecewiseConstantBSL
export PiecewiseLinearOceanSurfaceBSL, ImposedBSL, CombinedBSL
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

# transitions.jl
export AbstractTransition, SharpTransition, SmoothTransition

# layering.jl
export AbstractLayering
export UniformLayering, ParallelLayering, EqualizedLayering, FoldedLayering
export get_layer_boundaries, interpolate2layers

# convolutions.jl
# export ConvolutionPlan, convo!, nextfastfft, _zeropad!, samesize_conv!
export gaussian_smooth, conv!, ConvolutionPlan, ConvolutionPlanHelpers
export samesize_conv_indices


# tools.jl
export GIATools
export AbstractFFTBackend, ComplexFFTBackend, RealFFTBackend

# state.jl
export CurrentState, ReferenceState, KinematicBSL

# snapshot.jl
export StateSnapshot, snapshot!, restore!

# io.jl
export NetcdfOutput, NativeOutput, write_nc!, write_out!
export PaddedOutputCrop, AsymetricOutputCrop

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export lon360tolon180
export meshgrid, kernelcollect

export get_quad_coeffs, get_r, gauss_distr, generate_gaussian_field
export uniform_ice_cylinder, stereo_ice_cylinder, stereo_ice_cap
export zeros, not, deviceinfo, cudainfo, kernelpromote, kernelzeros, on_host
# Re-exported so `backend = CPU()` works without `using KernelAbstractions`.
# Vendor backends (`CUDABackend`, `ROCBackend`, …) come from their own packages.
export CPU

# derivatives.jl
export update_second_derivatives!   #, dxx!, dyy!

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
export LaterallyConstantSeaSurface, LaterallyVariableSeaSurface, ImposedSeaSurface
export AbstractBSLFormalism, GoelzerBSLFormalism, AdhikariBSLFormalism
export AbstractBarystaticContribution
export AbstractVolumeContribution, GoelzerVolumeContribution, NoVolumeContribution
export AbstractAdjustmentContribution,
    GoelzerAdjustmentContribution, NoAdjustmentContribution
export AbstractDensityContribution, GoelzerDensityContribution, NoDensityContribution
export update_dz_ss!

# material.jl
export AbstractCalibration, NoCalibration, SeakonCalibration, apply_calibration!

export AbstractCompressibility,
    IncompressibleMantle, CompressibleMantle, apply_compressibility!

export AbstractViscosityLumping, TimeDomainViscosityLumping
export FreqDomainViscosityLumping, MeanViscosityLumping, MeanLogViscosityLumping
export get_effective_viscosity_and_scaling, green_viscous

export get_relaxation_time, get_relaxation_time_weaker, get_relaxation_time_stronger
export get_rigidity, get_shearmodulus, get_elastic_green, get_flexural_lengthscale
export absorption_band_density, fit_prony_series

# solidearth.jl
export SolidEarth
export AbstractLithosphere, AbstractMantle
export RigidLithosphere, LaterallyConstantLithosphere, LaterallyVariableLithosphere
export RigidMantle, RelaxedMantle, ViscousMantle, TransientCreepMantle
export BurgersMantle, ExtendedBurgersMantle
export AbstractLithosphereColumn,
    IncompressibleLithosphereColumn, CompressibleLithosphereColumn

# deformation.jl
export update_dudt!, update_deformation_rhs!, thinplate_horizontal_displacement
export update_elasticresponse!

# analytic solutions
export analytic_solution

# data loaders
export load_dataset, get_greenintegrand_coeffs
export load_wiens2022
export load_lithothickness_pan2022, load_logvisc_pan2022
export load_ice6gd
export load_spada2011, spada_cases
export load_latychev_test3, load_latychev2023_ICE6G

# simulation.jl
export SolverOptions, Simulation, run!, init_integrator
export update_diagnostics!, step!

# integrators.jl
export AbstractIntegrator,
    EulerIntegrator, BS3Integrator, Tsit5Integrator, RKCIntegrator, integrate            # `init_integrator` exported above (simulation.jl group)

# stability.jl
export stability_function, real_axis_stability_limit, stability_limit
export spectral_radius_estimate, simulation_rhs_probe
export analytic_lambda_bound, stiffness_report

include("plots.jl")

# inversion.jl
export InversionConfig, InversionData, InversionProblem, ParameterReduction

# inverse/ (new inversion API)
export AbstractDiffMode, TangentMode, AdjointMode
export AbstractObservable,
    VerticalUpliftObservable,
    VerticalUpliftRateObservable,
    RelativeSeaLevelObservable,
    Observation
export SimulatedObservable, attach_simobs!
export AbstractEncoding,
    Test1Encoding,
    Test2Encoding,
    EOFEncoding,
    AutoEncoding,
    VariationalAutoEncoding,
    nparams
# note: `reconstruct!` is already exported by inversion.jl (shared generic)
export AbstractRegularization,
    TikhonovReg,
    L2Reg,
    SurfaceSmoothnessReg,
    DecodedBounds,
    BoundedQuantity,
    Log10Viscosity,
    UpperMantleDensity,
    LithoDensity
export AbstractRegTarget, ThetaTarget, FieldTarget, SurfaceTarget
export AbstractRegOrder, Order0, Order1
export AbstractInversion, IceLoadInversion, ParameterInversion, loss
export gradient!, loss_and_gradient!, solve!
export AbstractLoss, DefaultLoss, misfit
export ForwardRecord, record_forward!, replay_interval!

end
