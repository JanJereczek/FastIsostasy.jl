module FastIsostasy

using LinearAlgebra
using StatsBase
using FFTW
using FastGaussQuadrature
using DSP
using CUDA
using OrdinaryDiffEq
using Reexport
@reexport using Interpolations

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
export ColumnChanges
export PrecomputedFastiso

# parameters.jl
export init_domain
export init_physical_constants
export init_multilayer_earth

# utils.jl
export years2seconds
export seconds2years
export m_per_sec2mm_per_yr

export dist2angle_stereographic

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
export get_integrated_loadresponse
export forwardstep_isostasy
export forward_isostasy!
export isostasy
export apply_bc!
export ice_load

# geoid.jl
export get_geoid_green
export update_columnchanges!
export compute_geoid_response
export get_load_change

end
