# FastIsostasy.jl

❄ *Fast and friendly glacial isostatic adjustment on CPU and GPU.*

FastIsostasy is a friendly and flexible software package for simulations of regional glacial isostatic adjustment (GIA) with laterally-variable mantle viscosity and lithospheric thickness. It is mainly adressed to ice-sheet modellers who seek for (1) a good representation of solid-Earth mechanics at virtually zero computational cost, (2) an approximation of the sea-level equation and (3) ready-to-use inversion tools to calibrate the model parameters to data. The simple interface of FastIsostasy allows to flexibly solve GIA problems within few lines of code. It is fully open-source under MIT license and was succesfully benchmarked against analytical, 1D GIA and 3D GIA model solutions.

FastIsostasy relies on a hybrid Fourier/finite-difference collocation of the problem introduced in [^Cathles1975] and solved in [^Lingle1985], [^Bueler2007]. Thanks to a simplification of the full problem from 3D to 2D space and the use of [optimized software packages](@ref Juliaecosystem), running kiloyears of regional GIA with $$\Delta x = \Delta y \simeq 45 \, \mathrm{km}$$ is a matter of seconds on a single CPU. For high resolution runs, the user can switch to GPU usage with minimal syntax change and enjoy the advantage of parallelization without requiring an HPC cluster. The central place of Fast-Fourier transforms in FastIsostasy's solving scheme inspired its name, along with a [GitHub repository](https://github.com/bueler/fast-earth)[^Bueler2007] that eased the first steps of this package. For GIA "purists", this package is likely to miss interesting processes but we belive that its ridiculous run-time can help to fast-prototype a problem before transfering it to a more comprehensive model.

!!! tip "Star us on GitHub!"
    If you have found this library useful, please consider starring it on [GitHub](https://github.com/JanJereczek/FastIsostasy.jl). This gives us a lower bound of the satisfied user count.

## Getting started

FastIsostasy.jl is a registered julia package. To install it, please run:

```julia
using Pkg
Pkg.add("FastIsostasy")
```

## How to read the docs?

If you want a quick introduction to GIA, please go to [A quick introduction to GIA](@ref). If you want to have a thorough but still accessbile introduction to GIA, we highly recommend reading [^Whitehouse2019]. If you want to get started right away, feel free to directly go to the [Examples](@ref). If you face any problem using the code or want to know more about the functionalities of the package, visit the [API reference](@ref). If you face a problem you cannot solve, please open a [GitHub issue](https://github.com/JanJereczek/FastIsostasy.jl/issues) with a minimal and reproduceable example. We also welcome feature requests!

## [Julia ecosystem](@id Juliaecosystem)

FastIsostasy.jl was written thanks to the sheer amount of work that people invested in the vast and well-documented Julia ecosystem. Major help from packages deserves major appreciation, in particular for:
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
- [DynamicalSystemsBase.jl](https://github.com/JuliaDynamics/DynamicalSystemsBase.jl)
- [DSP.jl](https://github.com/JuliaDSP/DSP.jl)
- [KalmanEnsembleProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
- [SpecialFunctions.jl]()
- And all the fantastic development tools that ease the every-day work so much: [Documenter.jl](), [Literate.jl](), [Test.jl]() as well as [Oceananigans]() for providing a template of how well geoscientific models can actually be documented.

[^Whitehouse2019]:
    Pippa Whitehouse et al. (2019): [Solid Earth change and the evolution of the Antarctic Ice Sheet](https://doi.org/10.1038/s41467-018-08068-y)

[^Cathles1975]:
    Lawrence Cathles (1985): Viscosity of the Earth's mantle

[^Lingle1985]:
    Lingle and Clark (1985): [A numerical model of interactions between a marine ice sheet and the solid earth: Application to a West Antarctic ice stream](https://doi.org/10.1029/JC090iC01p01100)

[^Bueler2007]:
    Ed Bueler et al. (2007): [Fast computation of a viscoelastic deformable Earth model for ice-sheet simulations](https://doi.org/10.3189/172756407782871567)
