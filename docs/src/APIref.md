# [API reference](@id api_ref)

## Utils

```@docs
    meshgrid
    init_domain
    get_differential_fourier
    init_physical_constants
    init_solidearth_params
    get_viscosity_ratio
    three_layer_scaling
    get_r
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
    forward_isostasy!
    forwardstep_isostasy
    cranknicolson_viscous_response
    euler_viscous_response
    apply_bc
    get_differential_fourier
    get_cranknicolson_factors
    plan_twoway_fft
    compute_elastic_response
```