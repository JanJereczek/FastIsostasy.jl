# Introduction

## Getting started

FastIsostasy.jl is work under devlopment and is not a registered julia package yet. To install it, please run:

```julia
using Pkg
Pkg.add("https://github.com/JanJereczek/FastIsostasy.jl")
```
## FastIsostasy.jl -- For whom?

This package is mainly addressed to ice sheet modellers looking for a regional model of glacial isostatic adjustment (GIA) that (1) captures the 3D structure of solid-Earth parameters, (2) computes an approximation of the sea-level equation, (3) runs kiloyear simulations on high resolution within minutes (without the need of HPC hardware) and (4) comes with ready-to-use calibration tools. For GIA "purists", this package is likely to miss interesting processes but we belive that the ridiculous run-time of FastIsostasy.jl can help them to perform some fast prototypting of a problem they might then transfer to a more comprehensive model.

!!! tip "Star us on GitHub!"
    If you have found this library useful, please consider starring it on [GitHub](https://github.com/JanJereczek/FastIsostasy.jl). This gives us a lower bound of the satisfied user count.

## How to read the docs?

If you already know about GIA, skip to [Overview of GIA for ice-sheet simulation](@ref). If you are already familiar with the complexity range of GIA models, skip to [Why FastIsostasy?](@ref). If you want to have a more thorough but still very accessbile introduction to GIA, we highly recommend reading [^Whitehouse2019]. If you want to get started right away, feel free to directly go to the [Examples](@ref). If you face any problem using the code or want to know more about the functionalities of the package, visit the [API reference](@ref). If you face a problem you cannot solve, please open a [GitHub issue]() with a minimal and reproduceable example.

## What is glacial isostatic adjustment?

The evolution of cryosphere components leads to changes in the ice and liquid water column and therefore in the vertical load applied upon the solid Earth. Glacial isostatic adjustment (GIA) denotes the mechanical response of the solid Earth, which is characterized by its vertical and horizontal displacement. GIA models usually encompass related processes, such as the resulting changes in sea-surface height and sea level.

The magnitude and time scale of GIA depends on the applied load and on solid-Earth parameters, here assumed to be the density, the viscosity and the lithospheric thickness. These parameters display a radial and sometimes also a lateral variability, further jointly denoted by parameter "heterogeneity". For further details, please refer to [^Wiens2021] and [^Ivins2023].

### Why should we care?

GIA is known to present many feedbacks on ice-sheet evolution. Their net effect is negative, meaning that GIA inhibits ice-sheet growth and retreat. In other words, it tends to stabilize a given state and is therefore particularly important in the context of paleo-climate and climate change.

The speed and magnitude of anthropogenic warming is a potential threat to the Greenland and the West-Antarctic ice sheets. They both represent an ice volume that could lead to multi-meter sea-level rise. The effect of GIA in this context appears to be particularly relevant - not only from a theoretical but also from a practical perspective, as a large portion of human livelihoods are concentrated along coasts.

## Motivation

### Overview of GIA for ice-sheet simulation

GIA models present a wide range of complexity, which can only be briefly mentioned here. On the lower end, models such as the Elastic-Lithopshere/Viscous-Asthenopshere are (1) cheap to run and (2) easy to implement, which has made them popular within the ice-sheet modelling community. They present some acceptable limitations such as (3) regionally approximating a global problem and (4) lacking the radially layered structure of the solid Earth. However, some limitations have shown to be too important to be overlooked -- mainly the fact that (5) the heterogeneity of the lithospheric thickness and upper-mantle viscosity cannot be represented.

On the higher end of the complexity spectrum, we find the 3D GIA models which address all the limitations of low-complexity models but are (1) expensive to run, (2) more tedious to couple to an ice-sheet model and (3) generally lack a well-documented and open-source code base. Due to these drawbacks, they do not represent a standard tool within the ice-sheet modelling community. Nonetheless, they are becoming increasingly used, as for instance in [^Gomez2018] and [^VanCalcar2023].

We here willingly omit to speak about 1D GIA models, as they lack the representation of heterogeneous solid-Earth parameters.

### Where is FastIsosatsy.jl on the complexity range?

Although they are increasingly being coupled to ice-sheet models, we believe that the expense of 3D GIA models can be avoided while still addressing the aforementioned limitations of simplistic models. Models specifically designed for ice-sheet modelling, such as [^Bueler2007] and [^Coulon2021], have shown first improvements in closing the gap between simplistic and expensive models. FastIsostasy continues this work by generalizing both of these contributions into one, while benchmarking results against 1D and 3D GIA models.

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

[^Whitehouse2019]:
    Pippa Whitehouse et al. (2019): [Solid Earth change and the evolution of the Antarctic Ice Sheet](https://doi.org/10.1038/s41467-018-08068-y)  (https://esurf.copernicus.org/articles/6/401/2018/)

[^Wiens2021]:
    Douglas Wiens et al. (2021): [The seismic structure of the Antarctic upper mantle](https://doi.org/10.1144/M56-2020-18)

[^Ivins2023]:
    Erik Ivins et al. (2023): [Antarctic upper mantle rheology](https://doi.org/10.1144/M56-2020-19)

[^Gomez2018]:
    Natalya Gomez et al. (2018): [A Coupled Ice Sheet-Sea Level Model Incorporating 3D Earth Structure: Variations in Antarctica during the Last Deglacial Retreat](https://doi.org/10.1175/JCLI-D-17-0352.1)

[^VanCalcar2023]:
    Caroline van Calcar et al. (2023): [Simulation of a fully coupled 3D GIA - ice-sheet model for the Antarctic Ice Sheet over a glacial cycle](https://doi.org/10.5194/egusphere-2022-1328)

[^Bueler2007]:
    Ed Bueler et al. (2007): [Fast computation of a viscoelastic deformable Earth model for ice-sheet simulations](https://doi.org/10.3189/172756407782871567)

[^Coulon2021]:
    Violaine Coulon et al. (2021): [Contrasting Response of West and East Antarctic Ice Sheets to Glacial Isostatic Adjustment](https://doi.org/10.1029/2020JF006003)

[^Goelzer2020]:
    Heiko Goelzer et al. (2020): [Brief communication: On calculating the sea-level contribution in marine ice-sheet models](https://doi.org/10.5194/tc-14-833-2020)

[^Snyder1987]:
    John Snyder (1987): [Map projections -- A working manual](https://pubs.er.usgs.gov/publication/pp1395)
