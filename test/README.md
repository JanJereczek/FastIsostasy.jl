# Testing FastIsostasy.jl

FastIsostasy.jl is validated by severals tests:

1. Load = ice cylinder with R = 1000 km, H = 1 km; Viscosity and lithospheric thickness as in Bueler et al. (2007). Here we know the transient analytic solution and use it to check the basic functionality of FastIsostasy.jl.

2. Here 2 load cases (ice disc and ice cap) from the benchmark proposed in Spada et al. (2011) are computed. The viscosity is homogeneous in x, y but not in z (2 layers that are translated into channel and halfspace layers for FastIsostasy).

3. Same as the first test but with viscosity depending on r. This test is still hypothetical, as no analytical nor benchmark solution is known.

4. Not a test as such, rather check whether output makes sense and gain insight out of it. Disc load over viscosity field as provided by Wiens et al. (2021).

5. Same as previous test but with PD Antractica (or deglaciation? or glacial cycle?) as load and compare to some literature result.