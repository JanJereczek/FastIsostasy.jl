


## Simple load and geometry - DIY

Nonetheless, as any high-level convenience function, [`fastisostasy`](@ref) has limitations. An ice-sheet modeller typically wants to embed FastIsostasy within a time-stepping loop. This can be easily done by getting familiar with some intermediate-level functions. We here illustrate this by letting an ice cap grow over time. This growth is unphysical for the sake of keeping the example simple. 

```@example MAIN
W = 3000e3
n = 6
Omega = ComputationDomain(W, n)
c = PhysicalConstants()
p = LateralVariability(Omega)

R = 1000e3                  # ice disc radius (m)
H = 1000.0                  # ice disc thickness (m)

u_0, ue_0 = copy(Omega.null), copy(Omega.null)
fi = FastIso(Omega, c, p, t_Hice_snapshots, Hice_snapshots,
    t_eta_snapshots, eta_snapshots, interactive_geostate; kwargs...)
u = copy(u_0)

for t in 0.0:10.0:100.0
    # fi.Hice = 
    u, dudt, ue, geoid, sealevel = forward_isostasy(dt, t_out, u, fi, BS3(), false)
    println("t = $t,    u_max = $(maximum(u)),    dudt_max = $(maximum(dudt))")
end
```

## GIA following Antarctic deglaciation

We now want to provide a tough example that presents:
- a heterogeneous lithosphere thickness
- a heterogeneous upper-mantle viscosity
- various viscous channels
- a more elaborate load that evolves over time
- changes in the sea-level

For this we run a deglaciation of Antarctica, based on the ice thickness estimated in [GLAC1D]().

```@example MAIN
W = 3000e3      # (m) half-width of the domain
n = 7           # implies an NxN grid with N = 2^n = 128.
Omega = ComputationDomain(W, n)
c = PhysicalConstants()
```

## Inversion of solid-Earth parameters

FastIsostasy.jl relies on simplification of the full problem and might therefore need a calibration step to match the output of a 3D GIA model. By means of an unscented Kalman inversion, one can e.g. infer the appropriate effective upper-mantle viscosity based on the response of a 3D GIA model to a given load. Whereas this is know to be a tedious step, FastIsostasy is developped to ease the procedure by providing a convenience struct `Paraminversion` that can be run by:

```@example MAIN
W = 3000e3                  # half-length of the square domain (m)
Omega = ComputationDomain(W, n)
c = PhysicalConstants()

lb = [88e3, 180e3, 280e3, 400e3]
lv = get_wiens_layervisc(Omega)
p = LateralVariability(
    Omega,
    layer_boundaries = lb,
    layer_viscosities = lv,
)
ground_truth = copy(p.effective_viscosity)

R = T(2000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)
Hcylinder = uniform_ice_cylinder(Omega, R, H)
t_out = years2seconds.(0.0:1_000.0:2_000.0)

t1 = time()
results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver=BS3(), interactive_geostate=false)
t_fastiso = time() - t1
println("Took $t_fastiso seconds!")
println("-------------------------------------")

tinv = t_out[2:end]
Hice = [Hcylinder for t in tinv]
Y = results.u_out[2:end]
paraminv = ParamInversion(Omega, c, p, tinv, Y, Hice)
priors, ukiobj = perform(paraminv)
logeta, Gx, e_mean, e_sort = extract_inversion(priors, ukiobj, paraminv)
```
