module FastIsostasy

using Reexport
using LinearAlgebra
using StatsBase
using FFTW
using FastGaussQuadrature
@reexport using Interpolations
using DSP
using CUDA

include("utils.jl")
include("physics.jl")

# Write your package code here.
export init_domain
export init_physical_constants
export init_multilayer_earth

export years2seconds
export seconds2years
export m_per_sec2mm_per_yr

export convert2Array
export copystructs2cpu

export matrify_vectorconstant
export loginterp_viscosity
export get_rigidity
export get_r
export gauss_distr

export mixed_fdx
export mixed_fdy
export mixed_fdxx
export mixed_fdyy

export precompute_fastiso
export quadrature1D
export meshgrid
export get_quad_coeffs
export get_integrated_loadresponse
export forwardstep_isostasy
export forward_isostasy!

export ComputationDomain
export PhysicalConstants
export MultilayerEarth

end
