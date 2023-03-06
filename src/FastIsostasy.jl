module FastIsostasy

using LinearAlgebra
using StatsBase
using FFTW
using FastGaussQuadrature
using DSP
using CUDA
using OrdinaryDiffEq: ODEProblem, solve, OrdinaryDiffEqAlgorithm
using Reexport
@reexport using Interpolations
@reexport using OrdinaryDiffEq: Euler, BS3, Tsit5, TanYam7, Vern9, VCABM

# Euler, Midpoint, Heun, Ralston, RK4, OwrenZen3, OwrenZen4, OwrenZen5, DP5, RKO65,
# TanYam7, DP8, Feagin10, Feagin12, Feagin14, TsitPap8,  BS5, Vern6, Vern7, Vern8,
# KuttaPRK2p5(dt=), Trapezoid(autodiff = false), PDIRK44(autodiff = false)

include("structs.jl")
include("parameters.jl")
include("utils.jl")
include("derivatives.jl")
include("geoid.jl")
include("mechanics.jl")

# structs.jl
export ComputationDomain
export PhysicalConstants
export MultilayerEarth
export ColumnHeights
export PrecomputedFastiso
export GeoState

# parameters.jl
export init_domain
export init_physical_constants
export init_multilayer_earth

# utils.jl
export years2seconds
export seconds2years
export m_per_sec2mm_per_yr

export dist2angle_stereographic

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
export init_fastiso_results
export precompute_fastiso
export quadrature1D
export meshgrid
export get_quad_coeffs
export get_elasticgreen
export forward_isostasy
export apply_bc!
export ice_load

# geoid.jl
export get_geoidgreen
export update_loadcolumns!
export compute_geoid_response
export get_load_change

end
