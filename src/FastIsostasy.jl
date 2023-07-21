module FastIsostasy

using LinearAlgebra
using Statistics: mean
using Distributions: MvNormal
using JLD2: jldopen
using DelimitedFiles: readdlm
using Interpolations: linear_interpolation, Flat
using FFTW: fft, ifft, plan_fft, plan_ifft
using AbstractFFTs: Plan, ScaledPlan
using FastGaussQuadrature: gausslegendre
using DSP: conv
using CUDA: CuArray, CUFFT.plan_fft, CUFFT.plan_ifft, allowscalar
using OrdinaryDiffEq: ODEProblem, solve, OrdinaryDiffEqAlgorithm
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions

using Reexport
@reexport using Interpolations
@reexport using OrdinaryDiffEq: Euler, Midpoint, Heun, Ralston, BS3, BS5, RK4,
    OwrenZen3, OwrenZen4, OwrenZen5, Tsit5, DP5, RKO65, TanYam7, DP8,
    Feagin10, Feagin12, Feagin14, TsitPap8, Vern6, Vern7, Vern8, Vern9,
    VCABM, Rosenbrock23, QNDF, FBDF, ImplicitEuler
# KuttaPRK2p5(dt=), Trapezoid(autodiff = false), PDIRK44(autodiff = false)

include("structs.jl")
include("utils.jl")
include("derivatives.jl")
include("geostate.jl")
include("mechanics.jl")
include("inversion.jl")

# structs.jl
export ComputationDomain
export PhysicalConstants
export MultilayerEarth
export PrecomputedFastiso
export GeoState
export SuperStruct

# utils.jl
export years2seconds, seconds2years, m_per_sec2mm_per_yr
export meshgrid, dist2angulardist, latlon2stereo, stereo2latlon
export kernelpromote, convert2Array, copystructs2cpu

export matrify, matrify
export loginterp_viscosity
export get_rigidity
export get_r
export gauss_distr

export samesize_conv

export load_prem, compute_gravity, ReferenceEarthModel, maxwelltime_scaling!, compute_shearmodulus

# derivatives.jl
export mixed_fdx
export mixed_fdy
export mixed_fdxx
export mixed_fdyy
export get_differential_fourier

# mechanics.jl
export quadrature1D
export meshgrid
export get_quad_coeffs
export get_elasticgreen
export fastisostasy
export dudt_isostasy!
export dudt_isostasy_sparse!
export corner_bc!

# estimation.jl
export init_optim, integrated_rmse, ViscOptim, optimize_viscosity, Options

# geostate.jl
export update_geoid!, update_sealevel!, update_loadcolumns!
export columnanom_ice, columnanom_water, columnanom_mantle
export columnanom_load, columnanom_full, totalmass_anom
export get_loadchange, get_geoidgreen
export update_slc!, update_slc_pov!, update_slc_den!
export update_V_af!, update_slc_af!
export update_V_pov!, update_V_den!

# inversion.jl
export ParamInversion
export perform
export extract_inversion

end
