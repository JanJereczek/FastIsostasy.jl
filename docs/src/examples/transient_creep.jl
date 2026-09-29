#=
# Transient creep

[`ViscousMantle`](@ref) models steady-state (secondary) creep: a single viscosity,
one relaxation timescale per wavenumber. Real mantle rock also shows *transient*
(primary) creep, a faster and partly recoverable deformation in the first years to
decades after a load change. Laboratory torsion experiments see it, and so does
the geodetic response to tides, post-seismic flow and rapid ice loss.

[`TransientViscoElasticMantle`](@ref) captures it with the elastic spring, the
steady Maxwell dashpot and `N` Kelvin–Voigt branches (a Prony series) in series.
This page benchmarks it against the extended Burgers model (EBM) of
[ivins_notes_2021](@citet), whose Fig. 8 shows the subsidence beneath the centre of
a suddenly imposed water disc on an incompressible, homogeneous half-space.

## Setup

The reference curves were digitised from panel (a) of their Fig. 8: mantle
viscosity $$\eta = 2 \times 10^{20} \, \mathrm{Pa \, s}$$, shear modulus
$$\mu = 67 \, \mathrm{GPa}$$, $$\rho = 3380 \, \mathrm{kg \, m^{-3}}$$,
$$g = 9.8 \, \mathrm{m \, s^{-2}}$$, one Maxwell curve and four EBM curves
with relaxation strength $$\Delta \in \{1.2, 1.9\}$$ and high cut-off period
$$\tau_H \in \{7.14, 18.6\} \, \mathrm{yr}$$. Time is in years, displacement in
metres (negative downwards).
=#

using FastIsostasy, CairoMakie

ivins2021 = Dict(  #hide
    :maxwell => ([0.95, 5.15, 9.10, 13.20, 17.30, 21.50, 24.45, 28.60, 31.79, 34.80, 37.75, 41.94, 46.04, 50.14],  #hide
        [-0.947, -0.978, -1.008, -1.041, -1.069, -1.104, -1.124, -1.157, -1.180, -1.202, -1.227, -1.257, -1.284, -1.310]),  #hide
    :d12_t714 => ([0.10, 1.05, 2.05, 3.05, 4.10, 5.10, 6.05, 8.20, 11.15, 16.30, 21.45, 27.50, 33.65, 39.90, 45.04, 50.05],  #hide
        [-0.935, -1.373, -1.480, -1.547, -1.586, -1.616, -1.641, -1.676, -1.706, -1.747, -1.776, -1.804, -1.833, -1.863, -1.890, -1.916]),  #hide
    :d19_t714 => ([0.10, 1.00, 2.05, 3.10, 4.00, 5.15, 6.10, 8.10, 10.30, 12.20, 14.25, 16.40, 19.30, 22.45, 26.64, 30.70, 34.70, 39.80, 43.80, 49.90],  #hide
        [-0.935, -1.578, -1.729, -1.804, -1.853, -1.900, -1.918, -1.959, -1.984, -2.000, -2.016, -2.031, -2.045, -2.059, -2.078, -2.096, -2.110, -2.129, -2.149, -2.169]),  #hide
    :d12_t186 => ([0.00, 1.05, 2.05, 3.10, 4.10, 5.15, 7.15, 10.25, 14.20, 18.35, 22.45, 26.60, 29.60, 34.75, 38.80, 42.85, 49.00],  #hide
        [-0.941, -1.239, -1.343, -1.406, -1.459, -1.490, -1.549, -1.616, -1.673, -1.718, -1.751, -1.782, -1.800, -1.837, -1.857, -1.873, -1.908]),  #hide
    :d19_t186 => ([0.05, 1.05, 2.10, 3.00, 3.96, 5.05, 6.15, 8.25, 10.34, 12.25, 15.30, 18.40, 21.50, 24.55, 28.69, 33.75, 37.80, 41.90, 45.95, 50.10],  #hide
        [-0.933, -1.402, -1.545, -1.627, -1.686, -1.739, -1.780, -1.845, -1.886, -1.922, -1.963, -1.998, -2.022, -2.043, -2.067, -2.102, -2.118, -2.141, -2.159, -2.173]),  #hide
)  #hide
cases = [
    (key = :maxwell, label = "Maxwell"),
    (key = :d12_t714, label = "Δ = 1.2, τ_H = 7.14 yr", Δ = 1.2, τ_H = 7.14),
    (key = :d19_t714, label = "Δ = 1.9, τ_H = 7.14 yr", Δ = 1.9, τ_H = 7.14),
    (key = :d12_t186, label = "Δ = 1.2, τ_H = 18.6 yr", Δ = 1.2, τ_H = 18.6),
    (key = :d19_t186, label = "Δ = 1.9, τ_H = 18.6 yr", Δ = 1.9, τ_H = 18.6),
];

#=
Every curve starts with the same instantaneous elastic displacement, since the
rheology only matters for $$t > 0$$. Subtracting it isolates the time-dependent
part, which is what we compare below. It is the mean of the four EBM values at
$$t \approx 0$$ (the Maxwell curve has no sample there).
=#

w_elastic = sum(first(ivins2021[c.key][2]) for c in cases[2:end]) / 4
reference(key) = (ivins2021[key][1], ivins2021[key][2] .- w_elastic)

#=
The EBM uses a continuous absorption band of retardation times with exponent
$$\alpha = 1/2$$ and low cut-off $$\tau_L \to 0$$. We discretise it with
[`ExtendedBurgersMantle`](@ref), a Prony series of 6 log-spaced Kelvin branches
([`fit_prony_series`](@ref)), and couple the elastic spring in by converting it
into a [`TransientViscoElasticMantle`](@ref). The Maxwell curve is its
`N = 0` member, [`ViscoElasticMantle`](@ref).
=#

mu = 67e9
ebm(Δ, τ_H) = ExtendedBurgersMantle(shearmodulus = mu, relaxation_strength = Δ,
    alpha = 0.5, tau_L = 1e-4, tau_H = τ_H, nbranches = 6)
viscoelastic(c) = haskey(c, :Δ) ? TransientViscoElasticMantle(ebm(c.Δ, c.τ_H)) :
    ViscoElasticMantle(shearmodulus = mu)

#=
The half-space carries no plate, so the lithosphere is made vanishingly thin: the
flexural rigidity scales with the cube of its thickness. We impose the mantle
viscosity directly on the built `SolidEarth`, bypassing the viscosity lumping,
which otherwise returns an effective value for the layered profile. The
semi-implicit path discretises time itself and needs a fixed step. The elastic
diagnostics are refreshed every step, which is what [`TransientViscousMantle`](@ref)
needs (see the last section). The coupled mantles compute their elastic
displacement inside every step anyway.
=#

c = PhysicalConstants(g = 9.8)
domain = RegionalDomain(8e6, 7)                  # 16 000 km wide box, 128 × 128
dt = 0.1
t_out = [0.0; dt; 1.0:1.0:50.0]

function centre_displacement(mantle; radius, water)
    H = (water * c.rho_water / c.rho_ice) .* (domain.R .< radius)
    it = TimeInterpolatedIceThickness([0.0, 1e-3, 1e3], [zeros(domain), H, H], domain)
    se = SolidEarth(domain; lithosphere = LaterallyConstantLithosphere(),
        mantle = mantle, layer_boundaries = [1e3], layer_viscosities = [2e20],
        rho_uppermantle = 3380.0)
    se.effective_viscosity .= 2e20
    opts = SolverOptions(show_progress = false, integ = EulerIntegrator(dt = dt),
        dt_sparse_diagnostics = dt)
    nout = NativeOutput(vars = [:u, :ue], t = t_out, T = Float64)
    sim = Simulation(domain, BoundaryConditions(domain, ice_thickness = it),
        RegionalSeaLevel(), se, (0.0, t_out[end]); nout, opts, c)
    run!(sim)
    i, j = domain.nx ÷ 2, domain.ny ÷ 2
    u = [x[i, j] for x in sim.nout.vals[:u]]
    ue = [x[i, j] for x in sim.nout.vals[:ue]]
    ## Subtract the instantaneous elastic displacement, i.e. the first non-zero ue.
    ## It stays constant for the viscous family and relaxes for the viscoelastic one.
    ## The semi-implicit step freezes the load at the start of each step, so a load
    ## switched on at t = 0 acts from t = dt onwards: we shift time accordingly.
    ue0 = ue[findfirst(!iszero, ue)]
    ue = replace(ue, zero(ue0) => ue0)     # Farrell ue is zero before its first refresh
    return t_out[2:end] .- dt, (u .+ ue .- ue0)[2:end]
end

function plot_comparison(title, runs)
    fig = Figure(size = (860, 420))
    ax = Axis(fig[1, 1], xlabel = "time (yr)", ylabel = "viscous displacement (m)",
        title = title)
    for (k, (case, (t, w))) in enumerate(zip(cases, runs))
        color = Makie.wong_colors()[k]
        t_ref, w_ref = reference(case.key)
        scatter!(ax, t_ref, w_ref, color = color, markersize = 9)
        lines!(ax, t, w, color = color, linewidth = 2, label = case.label)
    end
    Legend(fig[1, 2], ax, framevisible = false)
    return fig
end

#=
## Matching the reference

The load that best explains all five digitised curves at once is a disc of radius
$$1050 \, \mathrm{km}$$ carrying $$0.677 \times 25 \, \mathrm{m}$$ of water. This
reproduces the digitised elastic offset and the Maxwell curve, and leaves the EBM
curves to test the rheology. With the load calibrated, FastIsostasy (lines)
reproduces the digitised curves (dots) closely.
=#

calibrated = [centre_displacement(viscoelastic(case), radius = 1.05e6,
    water = 0.677 * 25.0) for case in cases]
plot_comparison("Calibrated load: R = 1050 km, 0.677 × 25 m water", calibrated)

#=
Two ingredients matter. First, the elastic spring must be coupled to the viscous
elements, as discussed in the last section. Second, the absorption band needs a few
branches: a single Kelvin element at $$\tau_H$$ puts all its compliance at the
longest retardation time and makes the early subsidence far too slow, whereas
6 branches bring it within a few centimetres of the digitised curves.

## Load of the figure caption

The same runs with the load stated in the caption: a disc of radius
$$1750 \, \mathrm{km}$$ with $$25 \, \mathrm{m}$$ of water.
=#

caption = [centre_displacement(viscoelastic(case), radius = 1.75e6, water = 25.0)
    for case in cases]
plot_comparison("Caption load: R = 1750 km, 25 m water", caption)

#=
## Viscous vs. viscoelastic mantles

The mantle rheologies with a viscous element form two families, depending on
whether the unrelaxed elastic spring is coupled into the rheology:

| | spring handled apart (Farrell convolution) | spring coupled in series |
|---|---|---|
| steady creep | [`ViscousMantle`](@ref) | [`ViscoElasticMantle`](@ref) |
| transient creep | [`TransientViscousMantle`](@ref) | [`TransientViscoElasticMantle`](@ref) |

In the *viscous* family, the elastic displacement comes from the Farrell (1972)
Green's function of a layered Earth and is simply added to the viscous one. The
viscous displacement relaxes with a time $$2\eta k/\beta$$ per wavenumber $$k$$,
where $$\beta = \rho g + D k^4$$. This is the standard choice of FastIsostasy and
supports laterally variable parameters.

In the *viscoelastic* family, the spring sits in series with the viscous elements
of a homogeneous half-space. Per Fourier mode the response to a load $$F$$ is
$$F / (\beta + 2k\tilde\mu(s))$$ with the operational modulus

```math
\frac{1}{\tilde\mu(s)} = \frac{1}{\mu} + \frac{1}{\eta s} + \sum_j \frac{\Delta_j}{\mu (1 + s \tau_j)}.
```

The elastic displacement then relaxes as the viscous one grows, and the Maxwell
relaxation time becomes $$\eta/\mu + 2\eta k/\beta$$. In the semi-implicit step
this amounts to scaling load and restoring force by
$$\gamma = 2k\mu / (2k\mu + \beta)$$, with the elastic displacement
$$(F - \beta u)/(2k\mu + \beta)$$ computed alongside. This family is restricted
to laterally constant parameters.

On the calibrated load, the viscous family overshoots the reference, for steady and
transient creep alike:
=#

d19 = cases[3]
fig = Figure(size = (860, 420))
ax = Axis(fig[1, 1], xlabel = "time (yr)", ylabel = "viscous displacement (m)",
    title = "Calibrated load, Maxwell and Δ = 1.9, τ_H = 7.14 yr")
for (k, (case, label, mantle)) in enumerate([
        (cases[1], "ViscousMantle", ViscousMantle()),
        (cases[1], "ViscoElasticMantle", ViscoElasticMantle(shearmodulus = mu)),
        (d19, "TransientViscousMantle", ebm(d19.Δ, d19.τ_H)),
        (d19, "TransientViscoElasticMantle", TransientViscoElasticMantle(ebm(d19.Δ, d19.τ_H))),
    ])
    t, w = centre_displacement(mantle, radius = 1.05e6, water = 0.677 * 25.0)
    lines!(ax, t, w, color = Makie.wong_colors()[k], linewidth = 2, label = label,
        linestyle = isodd(k) ? :dash : :solid)
end
for case in (cases[1], d19)
    scatter!(ax, reference(case.key)..., color = :black, markersize = 9)
end
Legend(fig[1, 2], ax, framevisible = false)
fig

#=
The digitised curves are shown in black. The gap is not a matter of the Prony
series: it already shows up for steady creep, where there are no Kelvin branches.

!!! note "Choosing a family"
    The viscoelastic family is the right one for benchmarks against
    homogeneous-half-space solutions like the one above. It computes its own
    elastic displacement every step. Since the buoyancy of that deflection is
    already part of $$\beta$$, the lithospheric column anomaly
    $$\rho_\mathrm{litho} u^E$$ is not added to the load. The viscous family keeps
    the layered-Earth elastic response and laterally variable parameters. When
    combined with transient creep, set `dt_sparse_diagnostics = dt`: otherwise the
    elastic feedback on the load only changes every `dt_sparse_diagnostics`, and
    the Kelvin branches partly recover in between.

!!! warning "Current limitations"
    [`TransientViscousMantle`](@ref), [`ViscoElasticMantle`](@ref) and
    [`TransientViscoElasticMantle`](@ref) run on the semi-implicit path only:
    [`LaterallyConstantLithosphere`](@ref) or [`RigidLithosphere`](@ref),
    [`ComplexFFTBackend`](@ref) and a fixed step. Laterally variable parameters and
    the real-FFT backend raise an informative error.
=#
