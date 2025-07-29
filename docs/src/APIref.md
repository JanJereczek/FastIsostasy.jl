# API reference

## Simulation

```@docs
Simulation
SolverOptions
DiffEqOptions
run!
init_integrator
OrdinaryDiffEqTsit5.step!
```

## Computation domains

```@docs
AbstractDomain
RegionalDomain
GlobalDomain
```

## Boundary conditions

```@docs
BoundaryConditions
apply_bc!
```

### Ice thickness
```@docs
AbstractIceThickness
TimeInterpolatedIceThickness
ExternallyUpdatedIceThickness
```

### Boundary condition spaces
```@docs
AbstractBCSpace
RegularBCSpace
ExtendedBCSpace
```

### Boundary condition rules
```@docs
AbstractBC
OffsetBC
NoBC
CornerBC
BorderBC
DistanceWeightedBC
MeanBC
```

## Sea level

```@docs
SeaLevel
```

### Barystatic sea level (BSL)

```@docs
AbstractBSL
ConstantBSL
ConstantOceanSurfaceBSL
PiecewiseConstantBSL
ImposedBSL
CombinedBSL
update_bsl!
```

### Sea surface (gravitional response)

```@docs
AbstractSeaSurface
LaterallyConstantSeaSurface
LaterallyVariableSeaSurface
update_dz_ss!
```

### Sea level load

```@docs
AbstractSealevelLoad
NoSealevelLoad
InteractiveSealevelLoad
columnanom_water!
```

## Solid Earth

```@docs
SolidEarth
```

### Lithosphere

```@docs
AbstractLithosphere
RigidLithosphere
LaterallyConstantLithosphere
LaterallyVariableLithosphere
update_elasticresponse!
```

### Mantle

```@docs
AbstractMantle
RigidMantle
RelaxedMantle
MaxwellMantle
update_dudt!
```

### Layering
```@docs
AbstractLayering
UniformLayering
ParallelLayering
EqualizedLayering
FoldedLayering
get_layer_boundaries
```

### Calibration
```@docs
AbstractCalibration
NoCalibration
SeakonCalibration
apply_calibration!
```

### Viscosity lumping
```@docs
AbstractViscosityLumping
TimeDomainViscosityLumping
FreqDomainViscosityLumping
MeanViscosityLumping
MeanLogViscosityLumping
get_effective_viscosity_and_scaling
```

### Material utilities

```@docs
get_rigidity
get_shearmodulus
get_elastic_green
get_flexural_lengthscale
calc_viscous_green
get_relaxation_time
get_relaxation_time_weaker
get_relaxation_time_stronger
```

## Input/Output (I/O)
```@docs
load_dataset
NetcdfOutput
NativeOutput
```

## Makie utilities
```@docs
plot_transect
plot_load
plot_earth
plot_out_at_time
plot_out_over_time
```