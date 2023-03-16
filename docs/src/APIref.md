# [API reference](@id api_ref)

## Utils

```@docs
years2seconds
seconds2years
m_per_sec2mm_per_yr
matrify_vectorconstant
matrify_constant
get_r
meshgrid
dist2angulardist
scalefactor
latlon2stereo
stereo2latlon
ComputationDomain
get_differential_fourier
PhysicalConstants
get_viscosity_ratio
three_layer_scaling
get_greenintegrand_coeffs
build_greenintegrand
get_elasticgreen
get_quad_coeffs
quadrature1D
quadrature2D
get_normalized_lin_transform
normalized_lin_transform
```

## Physics

```@docs
    precompute_terms
    fastisostasy
    forwardstep_isostasy
    cranknicolson_viscous_response
    euler_viscous_response
    apply_bc
    get_differential_fourier
    get_cranknicolson_factors
    plan_twoway_fft
    compute_elastic_response
```