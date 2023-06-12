# API reference

## Basic structs

```@docs
ComputationDomain
PhysicalConstants
MultilayerEarth
RefGeoState
GeoState
PrecomputedFastiso
SuperStruct
FastisoResults
```

## Mechanics

```@docs
fastisostasy
forward_isostasy
init_results
forwardstep_isostasy!
dudt_isostasy!
simple_euler!
apply_bc
compute_elastic_response
```

## Sea-level

```@docs
update_geostate!
update_geoid!
get_loadchange
get_geoidgreen
update_loadcolumns!
update_sealevel!
update_slc!
update_V_af!
update_slc_af!
update_V_pov!
update_slc_pov!
update_V_den!
update_slc_den!
```

## Parameter inversion

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
sphericaldistance
scalefactor
latlon2stereo
stereo2latlon
get_rigidity
get_effective_viscosity
get_differential_fourier
get_viscosity_ratio
three_layer_scaling
loginterp_viscosity
hyperbolic_channel_coeffs
get_greenintegrand_coeffs
build_greenintegrand
get_elasticgreen
get_quad_coeffs
quadrature1D
quadrature2D
get_normalized_lin_transform
normalized_lin_transform
kernelpromote
```