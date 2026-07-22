# Public API

## Simulation

```@docs
Simulation
SolverOptions
run!
step!
init_integrator
PhysicalConstants
```

### Time integrators

See [Time integration](@ref) for the algorithms and how to choose between them.

```@docs
AbstractIntegrator
EulerIntegrator
BS3Integrator
Tsit5Integrator
RKCIntegrator
integrate
```

### Transitions

```@docs
AbstractTransition
SharpTransition
SmoothTransition
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
RegionalSeaLevel
```

### Barystatic sea level (BSL)

```@docs
AbstractBSL
ReferenceBSL
ConstantBSL
ConstantOceanSurfaceBSL
PiecewiseConstantBSL
ImposedBSL
CombinedBSL
AbstractBSLUpdate
InternalBSLUpdate
ExternalBSLUpdate
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

The mantle rheology says *what* is modelled. How the spectral step is computed is
the orthogonal FFT-backend axis below.

```@docs
AbstractMantle
RigidMantle
RelaxedMantle
ViscousMantle
TransientCreepMantle
BurgersMantle
ExtendedBurgersMantle
update_dudt!
```

### FFT backend

```@docs
AbstractFFTBackend
ComplexFFTBackend
RealFFTBackend
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
get_relaxation_time
get_relaxation_time_weaker
get_relaxation_time_stronger
absorption_band_density
fit_prony_series
```

## Input/Output (I/O)
```@docs
load_dataset
NetcdfOutput
NativeOutput
```