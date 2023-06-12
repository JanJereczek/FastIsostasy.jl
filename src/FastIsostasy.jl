module FastIsostasy

using LinearAlgebra
using Distributions: MvNormal
using Interpolations: linear_interpolation, Flat
using FFTW
using StatsBase
using FastGaussQuadrature
using DSP: conv
using CUDA
using CUDA: CuArray
using OrdinaryDiffEq: ODEProblem, solve, OrdinaryDiffEqAlgorithm
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions

using Reexport
@reexport using Interpolations
@reexport using OrdinaryDiffEq: Euler, BS3, Tsit5, TanYam7, Vern9, VCABM, Rosenbrock23, QNDF, FBDF, ImplicitEuler

# Euler, Midpoint, Heun, Ralston, RK4, OwrenZen3, OwrenZen4, OwrenZen5, DP5, RKO65,
# TanYam7, DP8, Feagin10, Feagin12, Feagin14, TsitPap8,  BS5, Vern6, Vern7, Vern8,
# KuttaPRK2p5(dt=), Trapezoid(autodiff = false), PDIRK44(autodiff = false)

include("structs.jl")
include("utils.jl")
include("derivatives.jl")
include("geostate.jl")
include("mechanics.jl")
include("inversion.jl")

# structs.jl
export XMatrix
export ComputationDomain
export PhysicalConstants
export MultilayerEarth
export PrecomputedFastiso
export GeoState
export SuperStruct

# utils.jl
export years2seconds
export seconds2years
export m_per_sec2mm_per_yr

export meshgrid
export dist2angulardist
export latlon2stereo
export stereo2latlon

export kernelpromote
export convert2Array
export copystructs2cpu

export matrify_vectorconstant
export loginterp_viscosity
export get_rigidity
export get_r
export gauss_distr

# derivatives.jl
export mixed_fdx
export mixed_fdy
export mixed_fdxx
export mixed_fdyy

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
export columnanom_load, columnanom_full, loadanom_green
export get_loadchange, get_geoidgreen
export update_slc!, update_slc_pov!, update_slc_den!
export update_V_af!, update_slc_af!
export update_V_pov!, update_V_den!

# inversion.jl
export ParamInversion
export perform
export extract_inversion

end
