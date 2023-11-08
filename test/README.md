# Testing FastIsostasy.jl

FastIsostasy.jl is validated by severals tests, which are required to run fast.
Their aim is to detect large errors, not to test the highest level of accuracy.
Therefore, we use a standard resolution of n = 6, Nx = Ny = 64, dx = dy ≈ 100km.

We here make a text explanation of the different set-ups for the automated tests:

1. Load = ice cylinder with R = 1000 km, H = 1 km; Viscosity and lithospheric thickness as in Bueler et al. (2007). Here we know the transient analytic solution and use it to check the basic functionality of FastIsostasy.jl. Also, check the multilayer solution.

2. Here 2 load cases (ice disc and ice cap) from the benchmark proposed in Spada et al. (2011) are computed. The viscosity is homogeneous in x, y but not in z (2 layers that are translated into channel and halfspace layers for FastIsostasy).

3. Test a domain with discontinuous lateral variability of lithopsheric thickness and upper-mantle viscosity.

4. Simulate glacial cycle.

5. Recover known parameter field with UKI