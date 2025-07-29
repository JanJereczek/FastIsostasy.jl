# API reference

## Simulation

```@docs
Simulation
run!
step!
```

## Computation domains

```@docs
RegionalDomain
```

## Boundary conditions

```@docs
BoundaryConditions
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
PiecewiseLinearBSL
ImposedBSL
CombinedBSL
```

### Sea surface (gravitional response)

```@docs
AbstractSeaSurface
LaterallyConstantSeaSurface
LaterallyVariableSeaSurface
```

### Sea level load

```@docs
AbstractSealevelLoad
NoSealevelLoad
InteractiveSealevelLoad
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
```

### Mantle

```@docs
AbstractMantle
RigidMantle
RelaxedMantle
MaxwellMantle
```

### Layering
```@docs
AbstractLayering
UniformLayering
ParallelLayering
EqualizedLayering
FoldedLayering
```

### Calibration

### Viscosity lumping

### Material
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
NetcdfOutput
NativeOutput
```