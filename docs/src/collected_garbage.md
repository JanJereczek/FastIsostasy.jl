


<!-- 

Inferring the lithospheric thickness and the upper-mantle viscosity is all but trivial since ice sheets famously present difficult access to scientific missions, leading to a lack of seismologic stations compared to other regions of the world. The presence of few kilometers of ice between most stations and the bedrock is a further challenge. Although major advances have been recently made, Antarctica is therefore the region where solid-Earth parameters are worst-constrained.

 [ [1](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JC090iC01p01100?casa_token=OEMWq5llrv4AAAAA:ok6M08OGPEbkORk44DO2apRXUPo7GkQrl2iwclQXXs6laMyI644GI7_XoluKjKSxWiJLAP5r91uQLeI), [2](https://www.cambridge.org/core/journals/annals-of-glaciology/article/fast-computation-of-a-viscoelastic-deformable-earth-model-for-icesheet-simulations/C878DBDD01271F6EB7874C9C4125196C) ].

Compared to [1, 2], FastIsostasy.jl does not assume constant fields for parameters of the solid Earth. It thus offers an open-source and performant generalization of the original articles.
 This allows to transform the PDE describing the physics into an ODE and accelerate the computation, mainly due to the highly optimized functions available for fast-fourier transform (FFT).


 Computing the vertical displacement of the bedrock can be computed much more efficiently by relying on the precomputation of some terms and operations. In FastIsostasy.jl, this can be easily performed:

-->


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