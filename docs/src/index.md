# FastIsostasy.jl

❄ *Fast and flexible glacial isostatic adjustment on CPU and GPU.*

!!! warning "Upgrade to v2.0!"
    FastIsostasy.jl has been refactored to fix some performance issues, offer a more versatile API and propose new features. FastIsostasy v2.0 is currently being tested and will be out soon. We strongly encourage to download v2.0 by running:
    
    ```julia
    ] add https://github.com/JanJereczek/FastIsostasy.jl
    ```

    This version will soon be registered and available through the usual `Pkg.add("FastIsostasy")` command.


![GlacialCycle](assets/isl-ice6g-N=350.gif)

FastIsostasy is a collection of models to compute the regional glacial isostatic adjustment (GIA) resulting from changes in the surface load (ice, liquid water and sediments). It is:
- Accessible: you can set up complex simulations with only a few lines of code, as demonstrated for the case of the last glacial cycle.
- Flexible: you can easily permute parameters and modelling choices to play Earth System Modelling like it's lego.
- Performant: the results obtained only marginally differ from those obtained by 1D and 3D GIA models, while displaying a speed-up of 2 to 6 orders of magnitude.

!!! tip "Star us on GitHub!"
    If you have found this library useful, please consider starring it on [GitHub](https://github.com/JanJereczek/FastIsostasy.jl). This gives us a lower bound of the satisfied user count.

## Getting started

FastIsostasy.jl is a registered julia package. To install it, simply run:

```julia
using Pkg
Pkg.add("FastIsostasy")
```

## How to read the docs?

If you want a quick introduction to GIA, please go to [Quick intro to GIA](@ref). If you want to have a thorough but still accessbile introduction to GIA, we highly recommend reading [whitehouse-solid-2019](@citet). If you want to get started right away, feel free to directly go to the [Tutorial](@ref). If you face any problem using the code or want to know more about the functionalities of the package, visit the [API reference](@ref). If you face a problem you cannot solve, please open a [GitHub issue](https://github.com/JanJereczek/FastIsostasy.jl/issues) with a minimal and reproduceable example. We also welcome feature requests!

## How to cite?

Swierczek-Jereczek, J., Montoya, M., Latychev, K., Robinson, A., Alvarez-Solas, J., & Mitrovica, J. (2024). FastIsostasy v1.0 – a regional, accelerated 2D glacial isostatic adjustment (GIA) model accounting for the lateral variability of the solid Earth. *Geoscientific Model Development, 17*(13), 5263-5290.

## [Julia ecosystem](@id Juliaecosystem)

FastIsostasy.jl was written thanks to the sheer amount of work that people invested in the vast and well-documented Julia ecosystem. Major help from packages deserves major appreciation, in particular for:
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
- [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
- [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl)
- [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl)
- [Proj.jl]()
- And all the fantastic development tools that ease the every-day work so much: [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl), [Literate.jl](https://github.com/fredrikekre/Literate.jl), [Test.jl](https://github.com/JuliaLang/julia/tree/master/usr/share/julia/stdlib/v1.9/Test) and [DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl). [Oceananigans](https://github.com/CliMA/Oceananigans.jl) and [SpeedyWeather]() provided fantastic templates in terms of API, code structure and documentation.