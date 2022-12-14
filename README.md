# FastIsostasy

Glacial Isostatic Adjustment (GIA) denotes the evolution over time of the solid-Earth vertical displacement depending on the load applied on it. It is an important process for ice-sheet modelling, and more generally for Earth system modelling. FastIsostasy.jl performs the computation of this displacement based on a Fourier collocation method described in [1, 2]. This allows to transform the PDE describing the physics into an ODE and accelerate the computation, mainly due to the highly optimized functions available for fast-fourier transform (FFT).

Compared to [1, 2], FastIsostasy.jl does not assume constant fields for parameters of the solid Earth. It thus offers an open-source and performant generalization of the original articles.

## Getting started

FastIsostasy.jl is work under development and must be downloaded from GitHub to be used. It will hopefully become a registered julia package in future.

## A three-layer model

Let x, y be the coordinates spanning the projection of the Earth surface and z the depth coordinate. The present model assumes three layers over the z-dimension:
- The elastic lithosphere.
- A channel representing the upper mantle (usually displaying lower viscosity than the rest of the mantle).
- A half-space representing the rest of the mantle.
The two-layer model is a special case of this and can be obtained by setting the chennel parameters to be the same as the ones of the half space.

For solid-Earth parameters (viscosity, flexural rigidity... etc) that are constant over x and y, the algorithm asymptotically converges to the exact solution with increasing grid resolution. Otherwise, an approximation method is used.