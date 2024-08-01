# Quick intro to GIA

Glacial isostatic adjustment (GIA) denotes the crustal displacement that results from changes in the ice, liquid water and sediment columns, as well as associated changes in Earth's gravity and rotation axis, ultimately impacting the sea level. The magnitude and time scale of the deformational response depends on the applied load and on solid-Earth parameters, i.e. the mantle viscosity, the lithosphere thickness and their respective density. These parameters display a radial and sometimes also a lateral variability, further jointly denoted by parameter "heterogeneity". For further details, please refer to [wiens-seismic-2022](@citet) and [ivins-antarctic-2022](@citet).

## Why do we care?

GIA is known to present many feedbacks on ice-sheet evolution. Their net effect is negative, meaning that GIA inhibits ice-sheet growth and retreat. In other words, it tends to stabilize a given state and is therefore particularly important in the context of paleo-climate and climate change.

The speed and magnitude of anthropogenic warming is a potential threat to the Greenland and the West-Antarctic ice sheets. They both represent an ice volume that could lead to multi-meter sea-level rise. The effect of GIA in this context appears to be particularly relevant - not only from a theoretical but also from a practical perspective, as a large portion of human livelihoods are concentrated along coasts.

## GIA models for ice-sheet simulation

GIA models present a wide range of complexity, which can only be briefly mentioned here. On the lower end, models such as the Elastic-Lithopshere/Viscous-Asthenopshere are cheap to run and easy to implement, which has made them popular within the ice-sheet modelling community. They present some acceptable limitations such as regionally approximating a global problem and lacking the radially layered structure of the solid Earth. However, some limitations have shown to be too important to be overlooked:
1. The GIA response is independent of the load's wavelength.
2. The heterogeneity of the lithospheric thickness and upper-mantle viscosity cannot be represented.
3. Changes in sea-surface height due to changes in mass repartition are ignored.

On the higher end of the complexity spectrum, we find the 3D GIA models which address all the limitations of low-complexity models but are expensive to run, more tedious to couple to an ice-sheet model and generally lack a well-documented and open-source code base. Due to these drawbacks, they do not represent a standard tool within the ice-sheet modelling community. Although, they are becoming increasingly used, as for instance in [gomez-coupled-2018](@citet) and [van-calcar-simulation-2023](@citet), we believe that the expense of 3D GIA models can be avoided while still addressing the aforementioned limitations of simplistic models. Models specifically designed for ice-sheet modelling, such as [bueler-fast-2007](@citet) and [coulon-contrasting-2021](@citet), have shown first improvements in closing the gap between simplistic and expensive models. FastIsostasy continues this work by generalizing both of these contributions into one.

We here omit to speak about other GIA models, since they lack the representation of heterogeneous solid-Earth parameters.

## FastIsosatsy.jl in the model hierarchy

FastIsostasy is capable of regionally reproducing the behaviour of a 3D GIA model at a computational cost that is reduced by 3 to 5 orders of magnitude. It relies on LV-ELVA, a generalisation of [bueler-fast-2007, coulon-contrasting-2021](@citet), and on the Regional Sea-Level Model (ReSeLeM).

FastIsostasy heavily relies on the Fast-Fourier Transform (FFT), as (1) its central PDE is solved by applying a Fourier collocation scheme and (2) important diagnostic fields are computed by matrix convolutions which can famously be accelerated by the use of FFT. FFT therefore inspired the name "FastIsostasy", along with a [GitHub repository](https://github.com/bueler/fast-earth) that eased the first steps of this package. The use of a performant language such as julia, as well as supporting performance-relevant computations on GPU allows FastIsostasy to live up to the expectations of low computation time.

We believe that FastIsostasy drastically reduces the burdens associated with using a 3D GIA model while offering all the complexity needed for ice-sheet modelling. As targeted and efficient climate-change mitigation relies on a good representation of important mechanisms in numerical models, we believe that this can be a significant contribution for future research.