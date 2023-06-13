# FastIsostasy

[![](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://janjereczek.github.io/FastIsostasy.jl/dev/)

## Getting started

FastIsostasy.jl is work under devlopment and is not a registered julia package yet. To install it, please run:

```julia
using Pkg
Pkg.add("https://github.com/JanJereczek/FastIsostasy.jl")
```

## For whom?

This package is mainly addressed to ice sheet modellers looking for a regional model of glacial isostatic adjustment (GIA) that (1) captures the 3D structure of solid-Earth parameters, (2) computes an approximation of the sea-level equation, (3) runs kiloyear simulations on high resolution within minutes (without the need of HPC hardware) and (4) comes with ready-to-use calibration tools. For GIA "purists", this package is likely to miss interesting processes but we belive that the ridiculous run-time of FastIsostasy.jl can help them to perform some fast prototypting of a problem they might then transfer to a more comprehensive model.

## Example

![Deglaciation](docs/src/assets/loaduplift_isostate_N128.gif)

The animation above depicts:
- **(a)** The ice-load anomaly with respect to 30 kiloyears before present as reconstructed in [GLAC1D](https://www.physics.mun.ca/~lev/dataAccess.html). Thus essentially simulate the isostatic adjustment induced by the last deglaciation.
- **(b)** The displacement rate of the bedrock resulting from changes in the ice load.
- **(c)** The total displacement obtained by integrating the rate.

This computation is performed with a time step of $\mathrm{dt} = 1 \, \mathrm{yr}$ and $N_{x} = N_{y} = 128 $. Without any parallelization, it is a matter of minutes on a modern machine and only requires few lines of code. For more details, see the [docs](https://janjereczek.github.io/FastIsostasy.jl/dev/).