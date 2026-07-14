# Public API

## Simulation

```@docs
Simulation
SolverOptions
DiffEqOptions
run!
step!
init_integrator
PhysicalConstants
```

### Time integrators

See [Time integration](@ref) for the algorithms and how to choose between them.

```@docs
FIEuler
FIBS3
FITsit5
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

## AD and inversion

FastIsostasy can be differentiated with [Enzyme](https://enzyme.mit.edu/julia/)
and run *backwards*: given observations of the surface, infer the ice load or the
solid-Earth parameters that produced them. Loading `Enzyme` activates the AD
extension; reverse mode additionally needs `Checkpointing`, and [`solve!`](@ref)
needs `Optim`. Worked examples: [Inverse ice history](@ref),
[Inverse calibration](@ref) and [Full-field viscosity inversion](@ref).

Note that AD requires a fixed-step integrator ([`FIEuler`](@ref)) and a
[`SmoothTransition`](@ref).

### Inversion problems

```@docs
AbstractInversion
IceLoadInversion
ParameterInversion
loss
gradient!
loss_and_gradient!
solve!
```

### Differentiation modes

```@docs
AbstractDiffMode
TangentMode
AdjointMode
```

### Observables

```@docs
AbstractObservable
VerticalUpliftObservable
VerticalUpliftRateObservable
RelativeSeaLevelObservable
Observation
SimulatedObservable
attach_simobs!
```

### Encodings

```@docs
AbstractEncoding
nparams
reconstruct!
Test1Encoding
Test2Encoding
EOFEncoding
AutoEncoding
VariationalAutoEncoding
```

### Loss models

```@docs
AbstractLoss
DefaultLoss
misfit
```

### Regularization and bounds

```@docs
AbstractRegularization
TikhonovReg
L2Reg
SurfaceSmoothnessReg
DecodedBounds
AbstractRegTarget
ThetaTarget
FieldTarget
SurfaceTarget
AbstractRegOrder
Order0
Order1
BoundedQuantity
Log10Viscosity
UpperMantleDensity
LithoDensity
```

### State snapshots

```@docs
StateSnapshot
snapshot!
restore!
```

## Makie utilities
```@docs
plot_transect
plot_load
plot_earth
plot_out_at_time
plot_out_over_time
plot_computation_time
```