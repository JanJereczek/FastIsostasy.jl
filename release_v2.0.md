## What's new?

### Still pending before v2.0

- [ ] progressmeter
- [ ] add Aqua testing!
- [x] AD
- [ ] Burgers rheology
- [ ] Adhikari
- [ ] Restart files
- [ ] externalise the proj functionalities
- [x] Put time integration into dev notes (add the AD notes, the EBM explanation... etc)
- [ ] Fix "arraykernel::Any" You rightfully pointed out that "arraykernel::Any" in the definition of RegionalDomain leads to type instability. Please suggest a simple way to fix it (you can also give several options to choose from if you want).
- [x] forward and inverse problems split
- [ ] calibration can be split: observations vs. 3D model
- [x] increment collapse level

### Physics

1. Burgers rheology
2. Horizontal motion
3. Add formulation by Adhikari
4. Implicit time stepping when possible

### API

The most important change from v1.0 to v2.0 is the API refactor. This allows the user to be much more flexible in their code:
1. BCs (type e.g. corner and domain e.g. extended!)
2. SolidEarthModel (including Burgers!)
3. Projection
4. Layering
5. BSL
6. RegionalComputationDomain => prepare v3.0 which will include GlobalComputationDomain!
4. Native and NetCDF output are now separated in a cleaner way
5. NetCDF output is more flexible by relying on Dict
6. Restart files
7. Progress meter and colored output
8. MakieExt
9. LinearSolveExt
10. NonlinearSolveExt
11. Naming is now very close to that Oceananigans and SpeedyWeather
12. Replaced ParallelStencils with KernelAbstractions (smaller dep and allows more flexibility for future kernel writing)
13. Overall improvement of typing ==> more flexibility for user
14. New logo!
15. CUDA is now a weakdep
16. Removed dependencies: OrdinaryDiffEq, ParallelStencils, DynamicalSystems, CUDA, NLSolve, Proj
17. New dependencies: Crayons, DocStringExtensions, JLD2, KA

### Performance

In the background, many important changes:
1. custom integrators + don't store solution anymore
2. Significant improvement in convolution performance (and 0 mem alloc)
3. RFFT for the viscous displacement
4. Memory allocation in time loop is 0 (acceleration via ipc, itp and bcs)
5. t_ode is now stored
6. Types are loosened without impact on performmance
7.  Muladd used
8.  Recurrent division replaced by multiplication with inverse
9.  Higher order spatial derivatives
10. Kernelabstractions instead of Stencils

### AD

Initial prompt:

> I want to make FastIsostasy (this package) AD compatible. The first application I want to target is the joint estimation of the ice thickness and mantle viscosity field, for instance over a glacial cycle, to match the present-day uplift rates (and potentially other observables). It is worth noting that I might have a low order representation of the ice thickness field and effective mantle viscosity via the use of an encoder (to reduce the dimensionality of the problem). The second application is to calibrate FastIsostasy against a 3D GIA model. The targetted parameters for this are the L2-regularized mantle viscosity (scalar or 2D field, optionally encoded), the lithospheric scaling factor (scalar), the lithospheric and mantle densities (scalar). Optionally, I might also want to estimate the scaling factor R in Fourier space (2D field, optionally encoded). Do you see an easy way to do this? My main requirements:
> - Everything should still be GPU comptabile
> - No loss of performance when AD is off (forward problems). We are doing everything in place, is that a problem?
> - An API that clearly distinguishes the inverse problems, something like InverseIceThickness (application 1), and InverseParams (application 2)
> - The use of SparseConnectivityTracers.jl so that we save computation time
> - The use of DifferentiationInterface.jl so that we are flexible
> - A smart way of checkpointing

>Do you think that it's feasible?

Now we have a pretty solid base that still needs to give proofs on the two test experiments.

### Requirements to the group

- try simulate something
- read the docs
- try to extend the code
- try to run CPU and GPU