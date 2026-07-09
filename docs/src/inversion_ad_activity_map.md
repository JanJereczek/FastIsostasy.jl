# Enzyme activity map for `Simulation`

Reference for every `Enzyme.autodiff` call on the differentiated forward model
(roadmap §4). "Activity" = how each object is annotated when the whole
`Simulation` is threaded through Enzyme: `Const` (not differentiated, no shadow) vs
`Duplicated`/`BatchDuplicated` (carries a shadow that Enzyme reads/accumulates).

The differentiated quantity is `loss(prob, θ)`; θ enters the sim through
`reconstruct!`, which writes the decoded parameters into `solidearth` (viscosity,
densities) and/or the ice-load `TimeInterpolation2D`. Everything that θ can reach
through the forward model must be shadowed; everything else should be `Const` to
keep the tape small and correct.

## `Simulation` fields

| Field | Activity | Rationale |
|---|---|---|
| `domain` | **Const** | Grids, wavenumbers, quadrature — fixed geometry, θ-independent. |
| `c` (`PhysicalConstants`) | **Const** | Physical constants. |
| `bcs` | **Const** | After the `apply_bc!` `dot`-rewrite (no `buffer` mutation), `OffsetBC` holds only `space`/`x_border`/`W`, none θ-dependent. `TimeInterpolatedIceThickness` **is Const for `ParameterInversion`**, but for `IceLoadInversion` its `H_itp.X` snapshots are the control → that sub-object is **Duplicated** (θ written there by `reconstruct!`). |
| `sealevel` | **Const** | Configuration; state lives in `now`. |
| `solidearth` | **Duplicated** | Holds `effective_viscosity`, `rho_uppermantle`, `rho_litho`, `litho_rigidity`, `pseudodiff_scaling` — the `ParameterInversion` controls. Shadow needed even when only a subset is active (unused shadow components stay zero). |
| `opts` | **Const** | Solver options, tableau, `transition`, `dt_min`. The dt-sequence is frozen during AD (roadmap §1 adjoint strategy). |
| `tools` (`GIATools`) | **mixed** | FFT/convolution **plans are Const** (fixed linear operators; differentiated via custom `EnzymeRules`, not by tracing FFTW). `prealloc` buffers are scratch overwritten every RHS eval → **Duplicated** (shadow scratch), *not* Const. |
| `ref` (`ReferenceState`) | **Const** | Reference/IC fields (`z_b` used by the smoothness reg is read-only). |
| `now` (`CurrentState`) | **Duplicated** | The evolving state (`u`, `ue`, `dudt`, masks, column anomalies, scalar volumes, `z_bsl`, …). This is the primary integrand — all of it is shadowed, **including the scalar fields** (`V_af`, `V_pov`, `V_den`, `delta_V`, `z_bsl`) and `count_sparse_updates` (`Int`, inactive-by-type but must round-trip). |
| `ncout` / `nout` | **Const** | Output configs; the writers themselves are `EnzymeRules.inactive`. |
| `timer` | **Const** | Wall-clock instrumentation; `t_computation!` is `EnzymeRules.inactive`, `timer.t = t` is inactive-by-type. |

## Inactive functions (declared in `FastIsostasyEnzymeExt`)

`EnzymeRules.inactive` (no derivative contribution, no shadow bookkeeping):

- `t_computation!` — `time()` + `push!` to timer vectors (roadmap §2 audit).
- NetCDF writers / output affects (`write_nc!`, `nc_affect!`, `nout_affect!`) — live
  only in the driver, off the RHS, but marked inactive defensively.
- Printing / logging.

## Custom `EnzymeRules` (not traced through the library)

- **Planned FFT application** `mul!(Y, plan, X)` for raw `plan::Union{cFFTWPlan,
  rFFTWPlan}`: forward tangent = the *same* plan applied to the input tangent
  (`mul!(dY, plan, dX)`); the ransform is linear so the Jacobian is the operator
  itself. The rule is deliberately **narrowed to raw plans** — `AbstractFFTs.ScaledPlan`
  is *traced* (its `mul! = lmul!(scale, mul!(y, p.p, x))` routes the inner `mul!`
  through this rule), because a custom rule matching `ScaledPlan` trips Enzyme's
  `roots_activep` assertion.
- **`NormalizedPlan` (the inverse-plan fix).** `plan_ifft`/`plan_irfft` are
  `ScaledPlan`s whose `Float64` `scale` Enzyme spuriously activates when the plan
  lives inside the `make_zero`'d sim — corrupting the gradient. `choose_fft_plans` /
  `_plan_irfft` wrap the inverse in `NormalizedPlan{P,S}` (src/convolutions.jl): the
  raw unnormalized plan `P` + the exact scale `S` **as a type parameter** (compile-
  time constant, hence Enzyme-inactive). Numerically identical to `ScaledPlan`
  (bit-for-bit). Marked `inactive_type`.

## Notes

- `apply_bc!(u, OffsetBC)` is a plain **linear in-place** op after the `dot` rewrite;
  Enzyme differentiates it natively (no custom rule). `u` is `Duplicated`, `bc` is
  `Const`.
- `reset_state!(sim)` resets `now` to the IC at the start of every `loss` eval; under
  AD both the primal and shadow `now` must be reset (shadow → zero).
- `TangentMode` requires an encoding (low-dim θ); the full `Simulation` is seeded with
  `BatchDuplicated` shadows, one batch column per θ component chunk.
- **`gradient!(::TangentMode)` must use `Enzyme.set_runtime_activity(Forward)`** — the
  explicit `update_dudt!` reuses a preallocated complex buffer (`P.fftU`) as both the
  fft input and the ifft output; static activity analysis silently returns the wrong
  tangent there, runtime activity fixes it. Shadow via `Enzyme.make_zero(prob)`.
