#=
# [Barystatic sea level: two formalisms](@id sealevel_formalisms)

The barystatic sea level (BSL) is the globally averaged sea-level change caused by
water leaving or entering the ocean. FastIsostasy computes the contribution of the
regional domain to it once per `dt_sparse_diagnostics`, and offers two ways of doing
so. They are selected on a single axis of [`RegionalSeaLevel`](@ref):

```julia
sealevel = RegionalSeaLevel(formalism = GoelzerBSLFormalism())     # default
sealevel = RegionalSeaLevel(formalism = AdhikariBSLFormalism())
```

[`GoelzerBSLFormalism`](@ref) follows [goelzer_brief_2020](@citet) and differences
three *absolute* volumes — ice above floatation ``V_\mathrm{af}``, a
meltwater/seawater density correction ``V_\mathrm{den}``, and a potential ocean
volume ``V_\mathrm{pov}`` (off by default) — against their values at the previous
coupling interval.

[`AdhikariBSLFormalism`](@ref) follows [adhikari_kinematic_2020](@citet) and instead
forms one *incremental* per-cell field over each interval ``\Delta t``:

```math
\begin{aligned}
\Delta H_S &= \Delta H_M + \Delta H_V \\
\Delta H_M &= \Delta H \, \mathcal{L}(t)\mathcal{L}(t{+}\Delta t)
            + \Delta H_F \, [1 - \mathcal{L}(t)\mathcal{L}(t{+}\Delta t)] \\
\Delta H_V &= (1 - \rho_w/\rho_o) \, [\Delta H - \Delta H_F] \,
              [1 - \mathcal{L}(t)\mathcal{L}(t{+}\Delta t)]
\end{aligned}
```

with ``\Delta H`` the ice-thickness change, ``\Delta H_F`` the change in height above
floatation, and ``\mathcal{L}`` the land mask. ``\Delta H_M`` changes ocean **mass
and volume**, ``\Delta H_V`` changes ocean **volume only** — meltwater occupies
slightly more volume than the seawater it displaces. The BSL change is then
``-(\rho_i/\rho_w) \sum \Delta H_S A / A_\mathrm{ocean}``.

The land-mask product is the whole trick: it is 1 only where the cell was land at
*both* ends of the interval. That partitions the domain into three regimes:

| Regime | ``\mathcal{L}(t)\mathcal{L}(t{+}\Delta t)`` | Meaning | What counts |
|---|---|---|---|
| 1 | 1 | grounded (or ice-free land) at both times | all of ``\Delta H`` |
| 2 | 0 | grounding line migrated over the interval | ``\Delta H_F``, plus the volume excess |
| 3 | 0 | floating at both times | volume excess only (``\Delta H_F = 0``) |

Rather than argue about the two on paper, the rest of this page runs them side by
side on three minimal experiments.
=#

using FastIsostasy, CairoMakie

W, n = 1.5f6, 6
domain = RegionalDomain(W, n, correct_distortion = false)

#=
Both experiments below build two simulations that are identical in every respect
*except* the formalism, so any difference between them is attributable to the
bookkeeping alone.
=#

function build_pair(; z_b, ice_bc, mantle, litho, t_end, dt)
    map((GoelzerBSLFormalism(), AdhikariBSLFormalism())) do formalism
        bcs = BoundaryConditions(domain, ice_thickness = ice_bc)
        se = SolidEarth(domain; mantle = mantle, lithosphere = litho)
        ## One BSL update per coupling step, so `delta_V` can be read out per
        ## interval below.
        opts = SolverOptions(show_progress = false, dt_sparse_diagnostics = dt,
            integ = EulerIntegrator(dt = dt))
        sealevel = RegionalSeaLevel(formalism = formalism)
        Simulation(domain, bcs, sealevel, se, (0f0, t_end);
            opts = opts, z_b_ref = z_b)
    end
end

#=
`sim.now.delta_V` holds the volume withheld from the ocean over the interval that
just closed, so `-delta_V / A_ocean` is that interval's contribution to global-mean
sea level in metres. Stepping manually with [`init_integrator`](@ref) and
[`step!`](@ref) — the same idiom as the [coupling example](@ref coupling) — lets us
read it out once per interval.
=#

function bsl_history(sim; t_end, dt)
    integrator = init_integrator(sim)
    t, dgmsl = Float32[], Float32[]
    tt = 0f0
    while tt < t_end
        step!(integrator, dt, true)
        tt += dt
        push!(t, tt)
        push!(dgmsl, -sim.now.delta_V / sim.sealevel.bsl.ref.A)
    end
    return t, cumsum(dgmsl)
end

#=
## Experiment 1: they agree exactly when the relative sea level is frozen

A parabolic ice cap sits on a bed that slopes from `+400 m` at the centre down to
marine depths, and melts away over 2 kyr. With [`RigidMantle`](@ref),
[`RigidLithosphere`](@ref) and [`ConstantBSL`](@ref) neither the bedrock nor the sea
surface moves, so the relative sea level satisfies ``\Delta R = \Delta S - \Delta B
= 0``.
=#

z_b_slope = @. 400f0 - 1600f0 * (domain.R / 8f5)
ice_cap(scale) = @. scale * 2500f0 * sqrt(max(1 - (domain.R / 1f6)^2, 0))

t_melt = [0f0, 2f3]
melting_bc = TimeInterpolatedIceThickness(t_melt, [ice_cap(1f0), ice_cap(0f0)], domain)

frozen = build_pair(z_b = z_b_slope, ice_bc = melting_bc, mantle = RigidMantle(),
    litho = RigidLithosphere(), t_end = 2f3, dt = 50f0)
t1, gmsl_goelzer = bsl_history(frozen[1], t_end = 2f3, dt = 50f0)
_, gmsl_adhikari = bsl_history(frozen[2], t_end = 2f3, dt = 50f0)

#=
This is Case A1.1 of their Appendix A, and the agreement is not merely "to within the
density conversion" — the freshwater-vs-seawater factor cancels term by term between
``\Delta H_V`` and Goelzer's ``V_\mathrm{den}``, so the two are algebraically
identical here, in all three regimes. What is left is `Float32` roundoff accumulated
over the 40 intervals:
=#

maximum(abs, gmsl_goelzer .- gmsl_adhikari) / maximum(abs, gmsl_goelzer)

#-

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "Time (yr)", ylabel = "Cumulative GMSL contribution (m)",
    title = "Frozen relative sea level: the two formalisms coincide")
lines!(ax, t1, gmsl_goelzer, linewidth = 6, color = (:steelblue, 0.4),
    label = "GoelzerBSLFormalism")
lines!(ax, t1, gmsl_adhikari, linewidth = 2, color = :black, linestyle = :dash,
    label = "AdhikariBSLFormalism")
axislegend(ax, position = :lt)
fig

#=
## Experiment 2: they part ways as soon as the bedrock moves

Now a 1500 m ice sheet is grown on a shallow marine bed over 200 years and then
simply *held* for the remaining 4.8 kyr, while the default [`ViscousMantle`](@ref)
lets the bedrock subside under the new load. (The ice has to be grown rather than
imposed from the start: FastIsostasy works in anomalies relative to the reference
state, so an ice sheet that is present at `t = 0` carries no load anomaly and nothing
would deform.)

After year 200 no ice is exchanged with the ocean at all, so the true barystatic
contribution over that stretch is zero.
=#

z_b_shallow = fill(-200f0, domain)
H_const = 1500f0 .* (domain.R .< 8f5)
growth_bc = TimeInterpolatedIceThickness([0f0, 2f2, 5f3],
    [zeros(domain), H_const, H_const], domain)

sinking = build_pair(z_b = z_b_shallow, ice_bc = growth_bc, mantle = ViscousMantle(),
    litho = LaterallyConstantLithosphere(), t_end = 5f3, dt = 50f0)
t2, sink_goelzer = bsl_history(sinking[1], t_end = 5f3, dt = 50f0)
_, sink_adhikari = bsl_history(sinking[2], t_end = 5f3, dt = 50f0)

#=
Both formalisms agree that growing the ice sheet lowers global sea level. But once the
ice stops changing, `AdhikariBSLFormalism` reports ``\Delta H = 0`` and therefore no
further contribution — its curve is flat from year 200 on. `GoelzerBSLFormalism`
tracks ``H_\mathrm{af}``, which keeps shrinking as the bed sinks away beneath the ice,
and so reports an ongoing sea-level rise that no meltwater ever produced.

This is their Case A1.2 (their Fig. A1a), and it is the **only** place the two
formalisms differ in FastIsostasy: Regime 1 on marine bedrock with an evolving
relative sea level.
=#

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "Time (yr)", ylabel = "Cumulative GMSL contribution (m)",
    title = "Ice held constant after year 200, bed still sinking")
vlines!(ax, [2f2], color = (:black, 0.3), linestyle = :dot)
text!(ax, 3f2, -0.5f0, text = "ice constant from here on",
    align = (:left, :center), fontsize = 12)
lines!(ax, t2, sink_goelzer, linewidth = 3, color = :steelblue,
    label = "GoelzerBSLFormalism")
lines!(ax, t2, sink_adhikari, linewidth = 3, color = :black, linestyle = :dash,
    label = "AdhikariBSLFormalism")
axislegend(ax, position = :rb)
fig

#=
Over the 4.8 kyr of pure bedrock relaxation, the drift each path accumulates out of
nothing — about a metre of sea-level equivalent for Goelzer, identically zero for
Adhikari:
=#

sink_goelzer[end] - sink_goelzer[4], sink_adhikari[end] - sink_adhikari[4]

#=
For the Antarctic projections of their §3.3 this mass-component difference alone is
about 5 %, and the total difference between the methods 10–15 % over 350 years. A
further systematic 2–3 % separates the two conventions: Goelzer converts to seawater
equivalent and adds ``V_\mathrm{den}`` as a correction, whereas Adhikari converts to
**freshwater** equivalent throughout. The two decompositions are therefore not
comparable term by term, only in their sum.

## Experiment 3: the regime map

Selecting `AdhikariBSLFormalism` activates the [`KinematicBSL`](@ref) buffers on
[`CurrentState`](@ref), which expose the decomposition itself. Melting the ice cap of
Experiment 1 in a single 500-year interval lets us look at where each component lives.
(Under `GoelzerBSLFormalism` these arrays are zero-size and cost nothing.)
=#

retreat_bc = TimeInterpolatedIceThickness([0f0, 5f2],
    [ice_cap(1f0), ice_cap(0.6f0)], domain)
regime = build_pair(z_b = z_b_slope, ice_bc = retreat_bc, mantle = RigidMantle(),
    litho = RigidLithosphere(), t_end = 5f2, dt = 5f2)[2]

integrator = init_integrator(regime)
step!(integrator, 5f2, true)
k = regime.now.kinematic

#-

fig = Figure(size = (900, 320))
axopts = (; xlabel = "x (10³ km)", aspect = DataAspect())
x, y = domain.x ./ 1f6, domain.y ./ 1f6

ax1 = Axis(fig[1, 1]; title = "ΔH: ice-thickness change", axopts...)
hm1 = heatmap!(ax1, x, y, ice_cap(0.6f0) .- ice_cap(1f0), colormap = :dense)
Colorbar(fig[2, 1], hm1, vertical = false, label = "m")

ax2 = Axis(fig[1, 2]; title = "ΔH_M: ocean mass and volume", axopts...)
hm2 = heatmap!(ax2, x, y, k.delta_H_M, colormap = :dense)
Colorbar(fig[2, 2], hm2, vertical = false, label = "m")

ax3 = Axis(fig[1, 3]; title = "ΔH_V: ocean volume only", axopts...)
hm3 = heatmap!(ax3, x, y, k.delta_H_V, colormap = :amp)
Colorbar(fig[2, 3], hm3, vertical = false, label = "m")
fig

#=
Reading a radius outwards from the centre makes the partition explicit. Out to
``R \approx 700`` km the ice stays grounded — first on land, then on marine bedrock —
and ``\Delta H_M`` equals ``\Delta H`` to the last digit while ``\Delta H_V`` is zero:
Regime 1. In the ring near ``R \approx 750`` km the grounding line migrates over the
interval, and ``\Delta H_M`` is cut back from ``\Delta H = -661`` m to
``\Delta H_F = -417`` m, with the remainder showing up in ``\Delta H_V``: Regime 2.
Beyond it the ice floats at both times, ``\Delta H_F = 0`` kills ``\Delta H_M``
entirely, and only the volume excess survives: Regime 3. Their sum is what enters the
BSL.
=#

## The regime can be read straight off the decomposition, which is the whole point:
## ΔH_M == ΔH is Regime 1, ΔH_M == 0 is Regime 3, anything between is Regime 2.
regime_of(ΔH, ΔH_M) = ΔH_M ≈ ΔH ? 1 : (ΔH_M ≈ 0 ? 3 : 2)

let m = domain.my, ΔH = ice_cap(0.6f0) .- ice_cap(1f0)
    for i in (m, m + 8, m + 16, m + 18, m + 20)
        println("R = ", lpad(round(Int, domain.R[i, m] / 1f3), 4), " km",
            "   z_b = ", lpad(round(Int, z_b_slope[i, m]), 5), " m",
            "   ΔH = ", lpad(round(ΔH[i, m], digits = 1), 7),
            "   ΔH_M = ", lpad(round(k.delta_H_M[i, m], digits = 1), 7),
            "   ΔH_V = ", lpad(round(k.delta_H_V[i, m], digits = 1), 6),
            "   → Regime ", regime_of(ΔH[i, m], k.delta_H_M[i, m]))
    end
end

#=
## Caveats of the current implementation

- **The ocean mask is pointwise.** Their Eq. (3) repairs the level set by flipping any
  below-floatation region that is not connected to the global ocean — a continental
  trough, a subglacial basin, a proglacial lake — back to land. That repair is not
  implemented, so such regions count as ocean and ``H_F`` never takes the negative
  values their §3.1 admits. It also means Regimes 2 and 3 are algebraically identical
  between the two formalisms, which is why Experiment 2 had to reach for Regime 1.
- **The ocean area is the global hypsometric table** of [`ReferenceBSL`](@ref), not
  their ``A_\mathrm{ocean}(t + \Delta t)``. A regional-model stand-in, and the same
  one the Goelzer path uses.
- **The load is unchanged.** Their Eq. (14) loads the solid Earth with
  ``\rho_i \Delta H_M`` plus an induced ocean-load term, which is what closes the mass
  budget; here the ice load still comes from the full column anomaly. Use
  [`InteractiveSealevelLoad`](@ref) rather than [`NoSealevelLoad`](@ref) if the
  induced ocean load matters for your application.
- **The sea surface lags by one coupling interval.** The ``t`` and ``t + \Delta t``
  states of an interval are captured at the same point of the update sequence, before
  `z_ss` is refreshed — exactly as ``V_\mathrm{af}`` is in the Goelzer path — so the
  *difference* over the interval is unaffected.

## Copy-pastable code

```julia
using FastIsostasy, CairoMakie

W, n = 1.5f6, 6
domain = RegionalDomain(W, n, correct_distortion = false)

z_b = fill(-200f0, domain)
H_const = 1500f0 .* (domain.R .< 8f5)
## Grown over 200 yr, then held: the model works in anomalies relative to the
## reference state, so ice already present at t = 0 carries no load and nothing moves.
ice_bc = TimeInterpolatedIceThickness([0f0, 2f2, 5f3],
    [zeros(domain), H_const, H_const], domain)

function bsl_history(formalism; t_end = 5f3, dt = 50f0)
    bcs = BoundaryConditions(domain, ice_thickness = ice_bc)
    se = SolidEarth(domain)
    opts = SolverOptions(show_progress = false, dt_sparse_diagnostics = dt,
        integ = EulerIntegrator(dt = dt))
    sim = Simulation(domain, bcs, RegionalSeaLevel(formalism = formalism), se,
        (0f0, t_end); opts = opts, z_b_ref = z_b)

    integrator = init_integrator(sim)
    t, dgmsl, tt = Float32[], Float32[], 0f0
    while tt < t_end
        step!(integrator, dt, true)
        tt += dt
        push!(t, tt)
        push!(dgmsl, -sim.now.delta_V / sim.sealevel.bsl.ref.A)
    end
    return t, cumsum(dgmsl)
end

t, goelzer = bsl_history(GoelzerBSLFormalism())
_, adhikari = bsl_history(AdhikariBSLFormalism())

fig, ax, _ = lines(t, goelzer, label = "GoelzerBSLFormalism")
lines!(ax, t, adhikari, linestyle = :dash, label = "AdhikariBSLFormalism")
axislegend(ax)
fig
```
=#
