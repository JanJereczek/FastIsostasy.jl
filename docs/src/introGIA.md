# Quick intro to GIA

Glacial isostatic adjustment (GIA) is the response of the solid Earth — together
with its gravity field, its oceans and even its rotation axis — to the
redistribution of mass at its surface as ice, liquid water and sediments come and
go. When an ice sheet grows, its weight pushes the ground down; when it melts, the
ground slowly springs back up. The magnitude and, crucially, the *time scale* of
this response depend both on the load that is applied and on the properties of the
solid Earth: the mantle viscosity, the lithosphere thickness and their respective
densities. These parameters vary with depth (radially) and, in many regions, also
from place to place (laterally) ([wiens_seismic_2022](@citet);
[ivins_antarctic_2022](@citet)).

If you are new to the field and want a thorough but still accessible overview, the
review by [whitehouse_solid_2019](@citet) is an excellent starting point; the
sections below distil the ideas most relevant to using this package.

## The load: what drives GIA

The "load" is simply the weight sitting on top of the solid Earth, and any change
in it drives GIA. In a glacial context the dominant terms are:

- **Ice**: kilometre-thick ice sheets impose enormous stresses on the ground. A
  1 km column of ice weighs roughly as much as a 300 m column of rock.
- **Liquid water**: as ice melts, meltwater is redistributed across the global
  ocean, loading the sea floor and unloading the formerly glaciated regions.
- **Sediments**: erosion and deposition move mass around more slowly, and are
  usually a second-order effect.

Because ice and water exchange mass, GIA is intrinsically a *coupled* problem: you
cannot solve for the deformation of the land without also keeping track of where
the ocean water goes.

## How the solid Earth responds

The key to GIA is that the solid Earth is **viscoelastic**: it behaves like a
spring on short time scales and like an extremely viscous fluid on long ones. A
useful mental picture is memory foam. Press a finger into it and it deforms
instantly (the *elastic* response); keep pressing and it keeps sinking slowly
towards a new equilibrium (the *viscous* response); lift your finger and it
recovers, but only gradually.

- The **elastic** part is essentially instantaneous and reversible. It is what lets
  the lithosphere flex like a stiff plate over a load.
- The **viscous** part is delayed and controls the long tail of the adjustment. It
  reflects the slow flow of mantle rock over thousands of years.

The single most important control on the *time scale* is the **mantle viscosity**.
Where the mantle is stiff (high viscosity, of order ``10^{21}``–``10^{23}\,
\mathrm{Pa\,s}``), the ground rebounds over many millennia: Scandinavia and the
Hudson Bay region are still rising at roughly a centimetre per year in response to
the disappearance of ice-age ice sheets that vanished over ten thousand years ago.
Where the mantle is weak (low viscosity, down to ``10^{18}``–``10^{19}\,
\mathrm{Pa\,s}``, as beneath West Antarctica, Iceland or Patagonia), the response
is much faster — decades to centuries — so the ground can react on the same time
scale as the ice it supports [whitehouse_solid_2019](@cite). This distinction is
central: fast GIA can influence an evolving ice sheet *within our lifetimes*, not
just over deglacial history.

## The structure of the solid Earth

For GIA purposes the Earth is usually described as a layered body:

- a relatively cold, rigid **lithosphere** that flexes elastically. Its thickness
  ranges from a few tens of kilometres in tectonically active regions (again, West
  Antarctica) to well over 200 km beneath old, stable cratons (such as much of East
  Antarctica);
- a **mantle** beneath it that flows viscously, often subdivided into an upper and
  a lower mantle with different viscosities.

Radial profiles of density and elastic moduli are commonly taken from seismically
derived reference models such as PREM [dziewonski_preliminary_1981](@cite), while
viscosity is inferred more indirectly, e.g. from post-glacial rebound and mantle
convection studies [cathles_viscosity_1975](@cite). Real Earth structure is not
only radially layered but also **laterally variable**, and viscosity in particular
can change by several orders of magnitude over relatively short distances. Whether,
and how well, a GIA model captures this lateral variability is one of the main
factors distinguishing different modelling approaches (see below).

## More than deformation: gravity, sea level and rotation

GIA is often abbreviated **GRD**, for its *gravitational*, *rotational* and
*deformational* components — a reminder that the ground moving up and down is only
part of the story.

- **Deformation** is the vertical (and horizontal) motion of the solid surface.
- **Gravity**: mass redistribution changes Earth's gravity field, which reshapes
  the sea surface. A large ice sheet gravitationally *attracts* the surrounding
  ocean, piling water up against itself; when it melts, that pull weakens and the
  local sea surface *drops*, even as sea level rises far away. This gravitationally
  self-consistent pattern is the reason melting the West Antarctic Ice Sheet would
  raise sea level in the Northern Hemisphere by *more* than the global average.
- **Rotation**: redistributing mass also nudges Earth's rotation axis, feeding back
  weakly onto sea level.

Solving for how meltwater redistributes itself while accounting for the deforming
solid surface and the changing gravity field is the classic *sea-level equation*.
FastIsostasy focuses on the deformational response, approximates the gravitational
one and neglects rotation; see [swierczek-jereczek_fastisostasy_2024](@citet) for
exactly which components are modelled and how.

## Why do we care?

GIA presents many feedbacks on ice-sheet evolution [whitehouse_solid_2019](@cite),
and their net effect is *negative*: GIA tends to inhibit both ice-sheet growth and
retreat. Two mechanisms are especially important for marine ice sheets, whose beds
lie below sea level:

- **Bedrock uplift near the grounding line.** As a marine ice sheet thins and
  unloads its bed, the ground rebounds, making the water column shallower. Shallower
  water reduces the ice's tendency to float, which can slow or halt grounding-line
  retreat — a stabilising, negative feedback on the marine ice-sheet instability.
- **The gravitational fingerprint.** A shrinking ice sheet pulls the nearby sea
  surface down, again lowering the water depth at the grounding line and reinforcing
  the same stabilising effect.

Because these feedbacks are strongest where the mantle is weak — precisely under
the vulnerable West Antarctic Ice Sheet — they can act on the decade-to-century
time scales that matter for future projections
([gomez_coupled_2018](@citet); [coulon_contrasting_2021](@citet);
[book_stabilizing_2022](@citet); [van_calcar_simulation_2023](@citet)). A faithful
representation of GIA is therefore needed to obtain realistic paleoclimatic
reconstructions of ice sheets, and reliable projections of future sea-level rise
under anthropogenic warming — a question of direct concern given how much of
humanity lives along coasts.

## How is GIA modelled?

Two broad families of models have traditionally been used:

- **Spectral / normal-mode models** treat the Earth as spherically symmetric (1-D,
  radially layered) and are elegant and efficient, but by construction cannot
  represent lateral variations in viscosity or lithosphere thickness.
- **Finite-element models** can resolve fully 3-D Earth structure, including lateral
  variability, but at a considerable computational cost that makes long or repeated
  simulations expensive — a real obstacle when GIA must be coupled to an evolving
  ice sheet or explored across many parameter choices
  [kachuck_rapid_2020](@cite).

FastIsostasy sits between these extremes. It captures the laterally variable
lithosphere thickness and mantle viscosity that matter for regions like Antarctica,
yet runs orders of magnitude faster than a 3-D model by reducing the problem from
three dimensions to two and solving it with a hybrid Fourier / finite-difference
scheme [swierczek-jereczek_fastisostasy_2024](@cite). This makes it well suited to
fast prototyping, long paleo runs, coupling with ice-sheet models and
parameter/data-assimilation studies. The [Analytical benchmark](@ref) and the other
worked examples show how to set up a simulation in a few lines of code, and the
[Public API](@ref) documents the available options.
