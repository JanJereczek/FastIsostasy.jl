module FastIsostasy

using LinearAlgebra
using StatsBase
using FFTW
using FastGaussQuadrature
using Interpolations
using DSP

include("utils.jl")
include("physics.jl")

# Write your package code here.
export init_domain
export init_physical_constants
export init_solidearth_params

export get_r

export precompute_terms
export quadrature1D
export meshgrid
export forwardstep_isostasy
export forward_isostasy!

export ComputationDomain
export PhysicalConstants
export SolidEarthParams
export LocalFields

end
