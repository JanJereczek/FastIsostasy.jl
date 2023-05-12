# Introduction

If you already know about GIA, skip to ...

## What is GIA?

The evolution of cryosphere components leads to changes in the ice and liquid water columnand therefore in the vertical load applied upon the solid Earth. Glacial isostatic adjustment (GIA) denotes the mechanical response of the solid Earth, which is characteized by its vertical and horizontal displacement. GIA models usually encompass related processes, such as the resulting changes in sea-surface height and sea level.

## Why should you care?

GIA is known to present many feedbacks on ice-sheet evolution. Their net effect is negative, meaning that GIA inhibits ice-sheet growth and retreat. In other words, it tends to stabilize a given state.

## [Why FastIsosatsy.jl?](@ref motivation)




the evolution over time of the solid-Earth vertical displacement depending on the load applied on it. It is an important process for ice-sheet modelling, and more generally for Earth system modelling. `FastIsostasy.jl` performs the computation of this displacement based on a Fourier collocation method described in [ [1](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JC090iC01p01100?casa_token=OEMWq5llrv4AAAAA:ok6M08OGPEbkORk44DO2apRXUPo7GkQrl2iwclQXXs6laMyI644GI7_XoluKjKSxWiJLAP5r91uQLeI), [2](https://www.cambridge.org/core/journals/annals-of-glaciology/article/fast-computation-of-a-viscoelastic-deformable-earth-model-for-icesheet-simulations/C878DBDD01271F6EB7874C9C4125196C) ]. This allows to transform the PDE describing the physics into an ODE and accelerate the computation, mainly due to the highly optimized functions available for fast-fourier transform (FFT).
<!-- 
Compared to [1, 2], FastIsostasy.jl does not assume constant fields for parameters of the solid Earth. It thus offers an open-source and performant generalization of the original articles. -->


!!! tip "Star us on GitHub!"
    If you have found this library useful, please consider starring it on [GitHub](https://github.com/JanJereczek/FastIsostasy.jl).
    This gives us an accurate lower bound of the (satisfied) user count.

## [Getting started](@id getting_started)

`FastIsostasy.jl` is work under development and must be downloaded from GitHub to be used. It will hopefully become a registered julia package in the future.
<!-- 
## A three-layer model

Let x, y be the coordinates spanning the projection of the Earth surface and z the depth coordinate. The present model assumes three layers over the z-dimension:
- The elastic lithosphere.
- A channel representing the upper mantle, usually displaying strong variance of viscosity over x and y.
- A half-space representing the rest of the mantle, usually with small variance of viscosity over x and y.
The two-layer model is a special case of this and can be obtained by setting the channel parameters to be the same as the ones of the half space.

![Schematic representation of the three-layer model](assets/sketch_3layer_model.png) -->

## Arrays

Array dimensions correspond to the spatial dimension of the variable they describe.
If they evolve over time, they are stored as vector of arrays.
For instance, the vertical displacement of the bedrock is a 2D variable that evolves
over time. Therefore, it is stored in a `Vector{Matrix}`.