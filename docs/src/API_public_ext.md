# Externalised public API

This page lists the public API of externalised modules, which are not part of the core package. These modules are optional and can be used to extend the functionality of FastIsostasy.jl by `using pkg` with `pkg` the package that triggers the extension.

## Makie utilities
```@docs
plot_transect
plot_load
plot_earth
plot_out_at_time
plot_out_over_time
plot_computation_time
```

## AD and inversion

FastIsostasy can be differentiated with [Enzyme](https://enzyme.mit.edu/julia/)
and run *backwards*: given observations of the surface, infer the ice load or the
solid-Earth parameters that produced them. Loading `Enzyme` activates the AD
extension; reverse mode additionally needs `Checkpointing`, and [`solve!`](@ref)
needs `Optim`. Worked examples: [Inverse ice history](@ref),
[Inverse calibration](@ref) and [Full-field viscosity inversion](@ref).

Note that AD requires a fixed-step integrator ([`EulerIntegrator`](@ref)) and a
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