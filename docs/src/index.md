# Introduction

## Getting started

FastIsostasy.jl is work under devlopment and is not a registered julia package so far. To install it, please run:

```julia
using Pkg
Pkg.add("https://github.com/JanJereczek/FastIsostasy.jl")
```
## FastIsostasy.jl -- For whom?

This package is mainly addressed to ice sheet modellers looking for a regional model of glacial isostatic adjustment (GIA) that (1) captures the 3D structure of solid-Earth parameters, (2) computes an approximation of the sea-level equation, (3) runs kiloyear simulations on high resolution within minutes (without the need of HPC hardware) and (4) comes with ready-to-use calibration tools. For GIA "purists", this package is likely to miss interesting processes but we belive that the ridiculous run-time of FastIsostasy.jl can help them to perform some fast prototypting of a problem they might then transfer to a more comprehensive model.

!!! tip "Star us on GitHub!"
    If you have found this library useful, please consider starring it on [GitHub](https://github.com/JanJereczek/FastIsostasy.jl). This gives us a lower bound of the satisfied user count.

## How to read the docs?

If you already know about GIA, skip to [Overview of GIA for ice-sheet simulation](@ref). If you are already familiar with the complexity range of GIA models, skip to [Why FastIsostasy?](@ref). If you want to have a more thorough but still very accessbile introduction to GIA, we highly recommend reading [Whitehouse et al. 2018](https://esurf.copernicus.org/articles/6/401/2018/). If you want to get started right away, feel free to directly go to the [Examples](@ref). If you face any problem using the code or want to know more about the functionalities of the package, visit the [API reference](@ref). If you face a problem you cannot solve, please open a [GitHub issue]() with a minimal and reproduceable example.


## What is glacial isostatic adjustment?

The evolution of cryosphere components leads to changes in the ice and liquid water column and therefore in the vertical load applied upon the solid Earth. Glacial isostatic adjustment (GIA) denotes the mechanical response of the solid Earth, which is characterized by its vertical and horizontal displacement. GIA models usually encompass related processes, such as the resulting changes in sea-surface height and sea level.

The magnitude and time scale of GIA depends on the applied load and on solid-Earth parameters, here assumed to be the density, the viscosity and the lithospheric thickness. These parameters display a radial and sometimes also a lateral variability, further jointly denoted by parameter "heterogeneity". For further details, please refer to [Wiens et al. 2021](https://www.lyellcollection.org/doi/full/10.1144/M56-2020-18) and [Ivins et al. 2023](https://www.lyellcollection.org/doi/full/10.1144/M56-2020-19).

### Why should we care?

GIA is known to present many feedbacks on ice-sheet evolution. Their net effect is negative, meaning that GIA inhibits ice-sheet growth and retreat. In other words, it tends to stabilize a given state and is therefore particularly important in the context of paleo-climate and climate change.

The speed and magnitude of anthropogenic warming is a potential threat to the Greenland and the West-Antarctic ice sheets. They both represent an ice volume that could lead to multi-meter sea-level rise. The effect of GIA in this context appears to be particularly relevant - not only from a theoretical but also from a practical perspective, as a large portion of human livelihoods are concentrated along coasts.

## Motivation

### Overview of GIA for ice-sheet simulation

GIA models present a wide range of complexity, which can only be briefly mentioned here. On the lower end, models such as the Elastic-Lithopshere/Viscous-Asthenopshere are (1) cheap to run and (2) easy to implement, which has made them popular within the ice-sheet modelling community. They present some acceptable limitations such as (3) regionally approximating a global problem and (4) lacking the radially layered structure of the solid Earth. However, some limitations have shown to be too important to be overlooked -- mainly the fact that (5) the heterogeneity of the lithospheric thickness and upper-mantle viscosity cannot be represented.

On the higher end of the complexity spectrum, we find the 3D GIA models which address all the limitations of low-complexity models but are (1) expensive to run, (2) more tedious to couple to an ice-sheet model and (3) generally lack a well-documented and open-source code base. Due to these drawbacks, they do not represent a standard tool within the ice-sheet modelling community. Nonetheless, they are becoming increasingly used, as for instance in [Gomez et al. 2018](https://journals.ametsoc.org/view/journals/clim/31/10/jcli-d-17-0352.1.xml?tab_body=pdf) and [Van Calcar et al. 2023](https://egusphere.copernicus.org/preprints/2022/egusphere-2022-1328/).

We here willingly omit to speak about 1D GIA models, as they lack the representation of heterogeneous solid-Earth parameters.

### Where is FastIsosatsy.jl on the complexity range?

Although they are increasingly being coupled to ice-sheet models, we believe that the expense of 3D GIA models can be avoided while still addressing the aforementioned limitations of simplistic models. Models specifically designed for ice-sheet modelling, such as [Bueler et al. 2007](https://www.cambridge.org/core/journals/annals-of-glaciology/article/fast-computation-of-a-viscoelastic-deformable-earth-model-for-icesheet-simulations/C878DBDD01271F6EB7874C9C4125196C) and [Coulon et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JF006003), have shown first improvements in closing the gap between simplistic and expensive models. FastIsostasy continues this work by generalizing both of these contributions into one, while benchmarking results against 1D and 3D GIA models.

FastIsostasy heavily relies on the Fast-Fourier Transform (FFT), as (1) its central PDE is solved by applying a Fourier collocation scheme and (2) important diagnostic fields are computed by matrix convolutions which can famously be accelerated by the use of FFT. FFT therefore inspired the name "FastIsostasy", along with a [GitHub repository](https://github.com/bueler/fast-earth) that eased the first steps of this package. The use of a performant language such as julia, as well as supporting performance-relevant computations on GPU allows FastIsostasy to live up to the expectations of low computation time.

We believe that FastIsostasy drastically reduces the burdens associated with using a 3D GIA model while offering all the complexity needed for ice-sheet modelling. As targeted and efficient climate-change mitigation relies on a good representation of important mechanisms in numerical models, we believe that this can be a significant contribution for future research.

## Technical details

In case you wonder, FastIsostasy.jl:

- Takes all parameters in SI units. This might be made more flexible in future by the use of [Unitful.jl]().
- Relies on a regular, square grid as those typically used for finite-difference schemes.
- Has a hybrid approach to solving its underlying PDE: while some terms are evaluated by finite differences, the usual expense of such method is avoided by applying a Fourier collocation scheme.
- For now only supports square domains with the number of points being a power of 2. This can accelerate computations of FFTs but will be made more flexible in future work.

FastIsostasy.jl largely relies on following packages:
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- [DSP.jl](https://github.com/JuliaDSP/DSP.jl)
- [KalmanEnsembleProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)

## References

1. [Whitehouse et al. 2018](https://esurf.copernicus.org/articles/6/401/2018/)
3. [Wiens et al. 2021](https://www.lyellcollection.org/doi/full/10.1144/M56-2020-18)
4. [Ivins et al. 2023](https://www.lyellcollection.org/doi/full/10.1144/M56-2020-19).
1. [Gomez et al. 2018](https://journals.ametsoc.org/view/journals/clim/31/10/jcli-d-17-0352.1.xml?tab_body=pdf)
5. [Van Calcar et al. 2023](https://egusphere.copernicus.org/preprints/2022/egusphere-2022-1328/)
6. [Bueler et al. 2007](https://www.cambridge.org/core/journals/annals-of-glaciology/article/fast-computation-of-a-viscoelastic-deformable-earth-model-for-icesheet-simulations/C878DBDD01271F6EB7874C9C4125196C)
7. [Coulon et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JF006003)
