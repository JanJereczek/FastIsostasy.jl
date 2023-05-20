# API reference

## Basic structs

```@docs
ComputationDomain
PhysicalConstants
MultilayerEarth
RefSealevelState
SealevelState
PrecomputedFastiso
SuperStruct
FastisoResults
```

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

## Mechanics

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

## Sea-level

```@docs
update_slstate!
update_geoid!
get_loadchange
get_geoidgreen
update_loadcolumns!
update_sealevel!
update_slc!
update V_af!
update_slc_af!
update V_pov!
update_slc_pov!
update_V_den!
update_slc_den!
```

## Parameter inversion