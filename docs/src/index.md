# FastIsostasy.jl

❄ *Fast and friendly glacial isostatic adjustment on CPU and GPU.*

![GlacialCycle](assets/isl-ice6g-N=350.gif)

FastIsostasy is a friendly and flexible model that regionally computes the glacial isostatic adjustment (GIA) with laterally-variable mantle viscosity and lithospheric thickness. It is mainly adressed to ice-sheet modellers who seek for (1) a good representation of solid-Earth mechanics at virtually zero computational cost, (2) an approximation of the sea-level equation and (3) ready-to-use inversion tools to calibrate the model parameters to data. The simple interface of FastIsostasy allows to flexibly solve GIA problems within few lines of code. It is fully open-source under MIT license and was succesfully benchmarked against analytical, 1D GIA and 3D GIA model solutions.

FastIsostasy relies on a hybrid Fourier/finite-difference collocation of the problem introduced in [cathles-viscosity-1975](@cite) and solved in [lingle-numerical-1985](@cite), [bueler-fast-2007](@cite). Thanks to a simplification of the full problem from 3D to 2D space and the use of [optimized software packages](@ref Juliaecosystem), running kiloyears of regional GIA with \$\Delta x = \Delta y = 45 \, \mathrm{km}\$ is a matter of seconds on a single CPU. For high resolution runs, the user can switch to GPU usage with minimal syntax change and enjoy the advantage of parallelization without requiring an HPC cluster. For GIA "purists", this package is likely to miss interesting processes but we belive that its ridiculous run-time can help to fast-prototype a problem before transfering it to a more comprehensive model.

!!! tip "Star us on GitHub!"
    If you have found this library useful, please consider starring it on [GitHub](https://github.com/JanJereczek/FastIsostasy.jl). This gives us a lower bound of the satisfied user count.

## Getting started

FastIsostasy.jl is a registered julia package. To install it, please run:

```julia
using Pkg
Pkg.add("FastIsostasy")
```

## How to read the docs?

If you want a quick introduction to GIA, please go to [Quick intro to GIA](@ref). If you want to have a thorough but still accessbile introduction to GIA, we highly recommend reading [whitehouse-solid-2019](@cite). If you want to get started right away, feel free to directly go to the [Tutorial](@ref). If you face any problem using the code or want to know more about the functionalities of the package, visit the [API reference](@ref). If you face a problem you cannot solve, please open a [GitHub issue](https://github.com/JanJereczek/FastIsostasy.jl/issues) with a minimal and reproduceable example. We also welcome feature requests!

## [Julia ecosystem](@id Juliaecosystem)

FastIsostasy.jl was written thanks to the sheer amount of work that people invested in the vast and well-documented Julia ecosystem. Major help from packages deserves major appreciation, in particular for:
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
- [DynamicalSystemsBase.jl](https://github.com/JuliaDynamics/DynamicalSystemsBase.jl)
- [KalmanEnsembleProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
- [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl)
- [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl)
- And all the fantastic development tools that ease the every-day work so much: [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl), [Literate.jl](https://github.com/fredrikekre/Literate.jl), [Test.jl](https://github.com/JuliaLang/julia/tree/master/usr/share/julia/stdlib/v1.9/Test) and [DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl). Despite being quite unrelated, [Oceananigans](https://github.com/CliMA/Oceananigans.jl) was a great template of how well geoscientific models can be documented.