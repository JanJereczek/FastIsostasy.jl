module FastIsostasy

using FFTW, FastGaussQuadrature

include("constants.jl")
include("solidearth_params.jl")
include("utils.jl")
include("physics.jl")

# Write your package code here.
export init_physical_constants
export init_solidearth_params
export init_domain
export init_integrator_tools
export meshgrid
export forwardstep_isostasy
export forward_isostasy!

export DomainParams
export PhysicalConstants

end
