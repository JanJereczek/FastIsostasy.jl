# Fortran version

Since Julia does not allow compilation to binaries yet, some researchers might find it hard to couple [FastIsostasy.jl] to their favourite ice-sheet model. To tackle this, a [Fortran code](https://github.com/palma-ice/isostasy) is under current development. It however lacks some features, which are summarised below:
- Library of time-integration methods reduced to explicit Euler and RK4,
- Only fixed time stepping allowed,
- No inversion routines for the UKI,
- Computation on GPU not supported.

These discrepancies are unlikely to be gapped in the future, mostly because the Fortran ecosystem is less extensive than the Julia one.
