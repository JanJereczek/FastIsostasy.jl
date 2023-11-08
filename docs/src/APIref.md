# API reference

## Basic structs

```@docs
KernelMatrix
ComputationDomain
PhysicalConstants
LayeredEarth
RefGeoState
GeoState
FastIsoTools
FastIsoProblem
```

## Mechanics

```@docs
solve!(::FastIsoProblem)
init
step!
update_diagnostics!
dudt_isostasy!
update_elasticresponse!
```

## Parameter inversion

```@docs
InversionConfig
InversionData
InversionProblem
solve!(::InversionProblem)
```

## Convenience

```@docs
load_dataset
reinit_structs_cpu
write_out!
```