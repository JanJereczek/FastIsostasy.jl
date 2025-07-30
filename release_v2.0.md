# What's new?

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
7. CallBackSet for output
8. Progress meter and colored output
9. MakieExt
10. LinearSolveExt
11. NonlinearSolveExt
12. Naming is now very close to that Oceananigans and SpeedyWeather

### Automatic Differentiation
An other important change is the inclusion of AD capabilities!!!

### Performance

In the background, many important changes:
1. ODE.jl does not store solution anymore
2. Significant improvement in convolution performance (and 0 mem alloc)
3. RFFT for the viscous displacement
4. Memory allocation in time loop is 0 (acceleration via ipc, itp and bcs)
5. Dependencies to DynamicalSystems.jl have been removed
6. All solvers from ODEtsit.jl and loworderRK are available
7. Horizontal motion can be computed according to thin plate theory
8. t_ode is now stored
9. Types are loosened without impact on performmance
10. Muladd used
11. Recurrent division replaced by multiplication with inverse
12. Higher order spatial derivatives


# Requirements to the group
- try simulate something
- read the docs
- try to extend the code
- try to run CPU and GPU