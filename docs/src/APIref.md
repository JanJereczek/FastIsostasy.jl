# API reference

## Basic structs

```@docs
ComputationDomain
PhysicalConstants
SolidEarthParameters
ReferenceState
CurrentState
FastIsoTools
SolverOptions
OceanSurfaceChange
FastIsoProblem
```

## Mechanics

```@docs
solve!(::FastIsoProblem)
init
step!
update_diagnostics!
lv_elva!
update_elasticresponse!
```

## Parameter inversion

```@docs
InversionConfig
InversionData
InversionProblem
inversion_problem
ParameterReduction
print_inversion_evolution
extract_inversion
reconstruct!
extract_output
```

## Convenience

```@docs
load_dataset
reinit_structs_cpu
write_out!
```