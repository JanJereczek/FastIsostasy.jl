module FastIsostasy

using FFTW
using FastGaussQuadrature
using LinearAlgebra
using Interpolations
using DSP

include("utils.jl")
include("physics.jl")

# Write your package code here.
export init_domain
export init_physical_constants
export init_solidearth_params

export precompute_terms
export quadrature1D
export meshgrid
export compute_elastic_response
export compute_viscous_response
export compute_viscous_response_heterogeneous
export forwardstep_isostasy
export forward_isostasy!
export cyclic_conv
export get_radial_gaussian_means

export ComputationDomain
export PhysicalConstants
export SolidEarthParams
export LocalFields

end
