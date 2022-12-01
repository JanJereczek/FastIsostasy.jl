module FastIsostasy

using FFTW
using FastGaussQuadrature
using LinearAlgebra
using Interpolations
using DSP: conv

include("constants.jl")
include("solidearth_params.jl")
include("utils.jl")
include("physics.jl")

# Write your package code here.
export init_physical_constants
export init_solidearth_params
export init_domain
export init_integrator_tools
export quadrature1D
export compute_elastic_response
export forwardstep_isostasy
export forward_isostasy!

export ComputationDomain
export PhysicalConstants
export SolidEarthParams

end