# A quick introduction to GIA

The evolution of cryosphere components leads to changes in the vertical load applied upon the solid Earth, namely through changes of the ice, liquid water and sediment columns. Glacial isostatic adjustment (GIA) denotes the mechanical response of the solid Earth, which is characterized by its vertical and horizontal displacement. GIA models usually encompass related processes, such as the resulting changes in sea-surface height and the migration of shorelines.

The magnitude and time scale of GIA depends on the applied load and on solid-Earth parameters, i.e. the mantle viscosity, the lithosphere thickness and their respective density. These parameters display a radial and sometimes also a lateral variability, further jointly denoted by parameter "heterogeneity". For further details, please refer to [^Wiens2021] and [^Ivins2023].

## Why do we care?

GIA is known to present many feedbacks on ice-sheet evolution. Their net effect is negative, meaning that GIA inhibits ice-sheet growth and retreat. In other words, it tends to stabilize a given state and is therefore particularly important in the context of paleo-climate and climate change.

The speed and magnitude of anthropogenic warming is a potential threat to the Greenland and the West-Antarctic ice sheets. They both represent an ice volume that could lead to multi-meter sea-level rise. The effect of GIA in this context appears to be particularly relevant - not only from a theoretical but also from a practical perspective, as a large portion of human livelihoods are concentrated along coasts.

## Overview of GIA models for ice-sheet simulation

GIA models present a wide range of complexity, which can only be briefly mentioned here. On the lower end, models such as the Elastic-Lithopshere/Viscous-Asthenopshere are cheap to run and easy to implement, which has made them popular within the ice-sheet modelling community. They present some acceptable limitations such as regionally approximating a global problem and lacking the radially layered structure of the solid Earth. However, some limitations have shown to be too important to be overlooked:
1. The GIA response is independent of the load's wavelength.
2. The heterogeneity of the lithospheric thickness and upper-mantle viscosity cannot be represented.
3. Changes in sea-surface height due to changes in mass repartition are ignored.

On the higher end of the complexity spectrum, we find the 3D GIA models which address all the limitations of low-complexity models but are expensive to run, more tedious to couple to an ice-sheet model and generally lack a well-documented and open-source code base. Due to these drawbacks, they do not represent a standard tool within the ice-sheet modelling community. Although, they are becoming increasingly used, as for instance in [^Gomez2018] and [^VanCalcar2023], we believe that the expense of 3D GIA models can be avoided while still addressing the aforementioned limitations of simplistic models. Models specifically designed for ice-sheet modelling, such as [^Bueler2007] and [^Coulon2021], have shown first improvements in closing the gap between simplistic and expensive models. FastIsostasy continues this work by generalizing both of these contributions into one

We here omit to speak about other GIA models, since they lack the representation of heterogeneous solid-Earth parameters.

## Where is FastIsosatsy.jl on the complexity ladder?

, while benchmarking results against 1D and 3D GIA models.

FastIsostasy heavily relies on the Fast-Fourier Transform (FFT), as (1) its central PDE is solved by applying a Fourier collocation scheme and (2) important diagnostic fields are computed by matrix convolutions which can famously be accelerated by the use of FFT. FFT therefore inspired the name "FastIsostasy", along with a [GitHub repository](https://github.com/bueler/fast-earth) that eased the first steps of this package. The use of a performant language such as julia, as well as supporting performance-relevant computations on GPU allows FastIsostasy to live up to the expectations of low computation time.

We believe that FastIsostasy drastically reduces the burdens associated with using a 3D GIA model while offering all the complexity needed for ice-sheet modelling. As targeted and efficient climate-change mitigation relies on a good representation of important mechanisms in numerical models, we believe that this can be a significant contribution for future research.


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
