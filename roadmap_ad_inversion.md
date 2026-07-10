# Roadmap: AD-based inversion for FastIsostasy

Status: **design locked, implementation not started.**
This file is the single source of truth for the AD/inversion effort. Tick boxes as
work lands; append decisions to §1 when they are made. Sessions should resume by
reading §1 (decisions) and finding the first unticked box.

---

## 1. Locked design decisions

| Topic | Decision |
|---|---|
| AD engine | Pure **Enzyme.jl** (forward + reverse). No DifferentiationInterface, no SparseConnectivityTracers, no ForwardDiff.jl/ReverseDiff.jl. |
| First AD target | **Explicit path**: `MaxwellMantle` + `LaterallyVariableLithosphere` (true RHS, Swierczek-Jereczek 2024) — the most commonly used configuration. The semi-implicit laterally-constant path and `RealMaxwellMantle` are **deferred** (§9, Appendix A). |
| Stencils | **Dual path** (revised 2026-07-07 after the perf gate): plain `@inbounds` loops on CPU (`Matrix`), KernelAbstractions kernels on GPU (`CuMatrix`, via `get_backend`), selected by dispatch. `@turbo`/LoopVectorization dropped (Enzyme can't differentiate it); plain loops are ~8× faster than KA-on-CPU and Enzyme-legal. Originally "KA-only", reverted because KA-CPU made forward runs ~35–45% slower — see §2 perf gate. |
| FFTs | All planned transforms on differentiated paths applied as `mul!(y, plan, x)`: writes the preallocated destination `y` in place (no allocation) but **preserves the input** `x` — unlike `pfft! * x`, which overwrites `x`. Input preservation is what AD needs (primal values stay available for the reverse pass). Custom `EnzymeRules` for plan application. |
| Adjoint strategy | Discrete adjoint, step-by-step reverse sweep over `advance_step!`. dt-sequence recorded in forward pass and **frozen** during reverse (no differentiation of the PI controller). |
| Checkpointing | **Checkpointing.jl with periodic schedules** from the start. Checkpoint unit = one save/observation interval (count known upfront); adaptive substeps free inside an interval; substep states stored densely in memory during the reverse recomputation of one interval. |
| Ice control variable | **Ice thickness** `H(x, t)` (encoded), with (a) smooth positivity `H = smooth_relu(H_raw; ε)` and (b) **L2 smoothness regularization on the implied surface** `s = H + b` (bed `b` from the reference/modeled state). Surface-elevation control was rejected: over shelves/ocean, `s` has near-zero load sensitivity (floating ice loads only via displaced water) and `s − b` clamping creates dead zones — thickness + surface-smoothness reg gives the smoothness benefit without the shelf pathology. |
| Smoothing | Trait `SharpTransition()` (default, current behavior, zero cost) vs `SmoothTransition(ε)`: `max(0,x) → (x + √(x²+ε²))/2`, Heaviside → `½(1 + x/√(x²+ε²))`, ε in meters of Haf (~1–10 m). Masks become eltype-`T` arrays on the smooth path. See §10 (Appendix B). |
| Diff modes | Own structs `TangentMode` (Enzyme forward; requires an encoding) / `AdjointMode` (Enzyme reverse; full fields allowed) under `AbstractDiffMode`. They carry policy (encoding requirement, checkpoint schedule, tangent batch size) and keep the core Enzyme-free. |
| Inversion types | `AbstractInversion`; **`IceLoadInversion`** (application 1), `ParameterInversion` (application 2). |
| Observables | Lightweight tag types defining only `extract!`; generic `Observation{O}` wrapper bundles tag + sampling + times + data + noise `σ`. Active tags: `VerticalUpliftObservable`, `VerticalUpliftRateObservable`, `RelativeSeaLevelObservable`. `HorizontalDisplacementRateObservable` stays **undefined** for now (future work, §11). Loss = Σₒ ½‖(Gₒ(θ) − yₒ)/σₒ‖² + reg. |
| Obs sampling | **Grid indices** (CartesianIndex) in v1. Physical coords + differentiable bilinear interpolation after Test 2 (user will provide real coordinate observations then). |
| Obs extraction | **Vector-based, GPU-accelerated (2026-07-08).** Stored indices converted to **linear/global** indices at `Observation` construction; `extract!` is a KA gather kernel `y[i] = field[idx[i]]` over a 1D range, backend-dispatched via `get_backend` (same dual-path shape as `src/derivatives.jl`; CPU = plain indexed loop, GPU = kernel). Closes the §11 GPU-gather item. |
| Regularization | **One configurable Tikhonov regularizer (2026-07-08)** with two DOFs: a **target** (a decoded field via a `BoundedQuantity`-style selector, or a θ-subset) and an **order** — `0` = magnitude `‖·‖²` (with optional per-component weights, which retires the §11 large-magnitude/Float32 domination issue), `1` = **gradient** `‖∇·‖²` via the FD stencils. Gradient/smoothness is the common case; magnitude is the exception. `SurfaceSmoothnessReg` = `(order=1, surface)`, `L2Reg` = `(order=0)` — kept as thin constructors over the general type. |
| Bounds/priors | No priors on encoded θ (hard to specify). Bounds imposed on **decoded** quantities via differentiable soft hinge penalties: `λ_b Σ [smooth_relu(lo − d)² + smooth_relu(d − hi)²]` on decoded fields/scalars. Optim's Fminbox (θ-space bounds) available as a fallback but not the primary mechanism. |
| Optimizer | **Optim.jl** (L-BFGS) via a weakdep extension. Core exposes `loss_and_gradient!` so any optimizer works. |
| Weakdeps | `FastIsostasyEnzymeExt` (engines + EnzymeRules), `FastIsostasyCheckpointingExt` gated on `[Enzyme, Checkpointing]`, `FastIsostasyOptimExt`. EnsembleKalman path dropped for now. |
| Precision | Float64 default for inversion runs (long reverse accumulations); forward production runs may stay Float32. |
| Old inversion.jl | Superseded; do not match its structure. `ParameterReduction`/EKP ext untouched until the new API lands, then deprecated. |

---

## 2. Phase 0 — Forward-model prerequisites (AD-legal explicit path)

Goal: the explicit `MaxwellMantle` + `LaterallyVariableLithosphere` forward model
runs identically to today, but every operation on the differentiated path is
Enzyme-legal.

- [x] **Stencils: drop `@turbo`, KA on GPU** (2026-07-07; CPU path revised same day):
      removed all `@turbo` variants, the `FiniteDiffParams`/`FiniteDiffMethod`
      arbitrary-order machinery, and the redundant derivative/thinplate overrides in
      `ext/FastIsostasyCUDAExt.jl`; dropped `FiniteDifferences` and
      `LoopVectorization` from `Project.toml`. **Final design (dual path):**
      `src/derivatives.jl` has plain-`@inbounds` methods dispatched on `Matrix` (CPU)
      and generic KA methods (GPU/`CuMatrix`, via `get_backend`); the `Matrix`
      methods are strictly more specific so CPU never touches KA, and GPU needs no
      ext code. `thinplate_horizontal_displacement!` relaxed to `AbstractMatrix`.
      (Interim KA-only step was reverted after the perf gate — see that box.)
      Fixed two latent GPU bugs in passing:
      (a) the 7-arg `update_second_derivatives!` used the fused kernel and ignored
      `u2`/`u3` (wrong for the distinct-input call in `update_deformation_rhs!`) —
      now routes fused only through the single-input path; (b) the GPU
      `thinplate_horizontal_displacement!` passed `domain.Dx` to `dy!` instead of
      `domain.Dy`. Verified: precompiles, `test_derivatives` passes, and a
      laterally-variable Maxwell forward run gives correct subsidence (~267 m for a
      1 km cylinder). GPU rerun still pending (needs CUDA hardware).
- [x] **`mul!` planned FFTs on the explicit path** (2026-07-07): `choose_fft_plans`
      now builds out-of-place complex plans (`plan_fft`/`plan_ifft`, CPU and CUDA);
      the laterally-variable `MaxwellMantle` `update_dudt!` applies them via `mul!`
      with a staged complex buffer, preserving each transform's input. The deferred
      semi-implicit laterally-constant path shares the plans, so its FFT calls were
      also converted to `mul!` (physics unchanged, algebraically identical). Field
      names `pfft!`/`pifft!` kept despite no longer mutating (matches the existing
      RealMaxwell convention; rename is a later cosmetic cleanup). `ConvolutionPlan`
      already used `mul!`+rfft — untouched. Verified: lat-variable Maxwell forward
      run reproduces the pre-change result **exactly** (−267.131 m); full test suite
      passes (derivatives 3/3, integrators 31/31, convolutions, barystatic).
      **⚠ Pre-existing bug found (NOT caused by this work):** the semi-implicit
      `MaxwellMantle` path (laterally-constant / rigid lithosphere + `FIEuler`,
      i.e. the documented Bueler implicit benchmark) produces `NaN` on this branch.
      Confirmed by running the config on a clean `HEAD` worktree (baseline also
      NaNs) and on the untouched `RealMaxwellMantle` path. Tracked in §8/§11 —
      needs a separate fix; irrelevant to the explicit AD target.
- [x] **Smooth transitions** (Appendix B) (2026-07-07):
  - [x] `src/transitions.jl`: `AbstractTransition`, `SharpTransition` (default),
        `SmoothTransition{T}(ε)` + scalar primitives `srelu`/`snegrelu`/`sheaviside`.
        Stored on `SolverOptions` (now `SolverOptions{TR}`); threaded via
        `sim.opts.transition`.
  - [x] Applied (transition-dispatched, sharp = byte-identical) in
        `height_above_floatation`, `update_Haf!`, `update_maskgrounded!`/
        `get_maskgrounded`, `update_maskocean!`/`get_maskocean`, and both the
        `max(z_ss−z_b,0)` relu and the 1 m ice-thickness switch in `watercolumn!`
        (smoothed as a partition of unity). `min(z_b−z_ss,0)` also smoothed.
        Added `not(::AbstractFloat)=1−x` for float masks.
  - [x] Masks are eltype-`T` arrays on the smooth path (Bool/`BitMatrix` on sharp);
        the state's `B` type parameter picks this up automatically from the
        constructor.
  - [x] Verified: `SharpTransition` reproduces the pre-change result **exactly**
        (−267.1312 m); `SmoothTransition` converges O(ε) (ε=100/10/1/0.1 →
        max|Δu|=12.7/1.33/0.133/0.013 m); smooth masks + interactive sea level run
        clean with `Float32` masks; full suite passes.
- [x] **Ice-load control — `TimeInterpolation2D` Enzyme-legality** (2026-07-07):
      `interpolate!` (src/interpolations.jl) is Enzyme-legal as-is: plain in-place
      broadcast, no closures; all control flow (`searchsortedfirst`, `t in ti.t`,
      `min/max(ti.t)`) is on **time**, not on the differentiated ice values; and the
      value is linear in the snapshots `ti.X[i], ti.X[i+1]`. For inversion,
      `reconstruct!` writes θ into `ti.X` (a `Vector{Matrix}`), which Enzyme shadows.
      No change needed. (The actual `IceLoadInversion` control wiring is Phase 1.)
- [x] **Side-effect audit** (2026-07-07): the RHS = `update_diagnostics!` (passed to
      `init_fi` at integrators.jl:418) and its entire sub-chain (`update_bedrock!`,
      `columnanom_*!`, `update_elasticresponse!`, `internal_update_bsl!`,
      `update_dz_ss!`, masks, `update_dudt!`) are **free of I/O and printing**.
      Output (`nc_affect!`/`nout_affect!`, `write_nc!`) lives only in
      `advance_with_output!` in the driver. **One AD blocker found**: the last line
      of `update_diagnostics!` calls `t_computation!(sim.timer)`, which calls `time()`
      (wall clock) and `push!`es to `sim.timer.t_computation`/`t_vec`. Must be
      `EnzymeRules.inactive` (Phase 2) or lifted out of the RHS. `sim.timer.t = t`
      (scalar write) is benign/inactive. → **resolved (2026-07-08): `EnzymeRules.inactive`
      one-liner in the ext (not lifted — lifting would change *when* it fires,
      per-RHS-eval → per-accepted-step, altering what the timer measures); see §4.**
- [x] **RHS `u`-mutation audit** (2026-07-07): `update_diagnostics!` calls
      `apply_bc!(u, sim.bcs.viscous_displacement)` on its **input** `u`
      (simulation.jl:288). For the default `OffsetBC` this is a weighted-global-mean
      removal: `bc.buffer .= bc.W.*X; X .-= (sum(bc.buffer) − bc.x_border)` — it
      mutates both the input `u` and `bc.buffer`. In the RK path each stage rebuilds
      `utmp` from scratch, so cross-stage corruption doesn't occur, but the FSAL
      `ks[1]` is computed on `integ.u`, so the projection persists into the state
      (probably intentional: keeps `u` on the zero-mean manifold). Enzyme can
      differentiate input mutation, but "input is also output" complicates the
      activity map. **Resolved (2026-07-08):** option (a) — the projection is a plain
      linear in-place op Enzyme handles natively, no custom rule; `u` → `Duplicated`.
      To make `bc` `Const` (rather than shadow it), **drop `bc.buffer`** and compute
      the reduction as `dot(bc.W, X)` (== old `sum(bc.buffer)`, allocation-free) — `bc`
      then mutates nothing. Also removes a temp write from every RHS eval. The
      "input-is-output / persists via FSAL `ks[1]`" note is intended semantics
      (zero-mean manifold), not an Enzyme issue. See §4.
- [x] **Perf regression gate** (2026-07-07): benchmarked forward runs (AD off) vs. a
      `HEAD` worktree (`@turbo` + in-place FFT), lat-variable Maxwell, min over reps.
      The interim KA-only build regressed badly; the **dual path (adopted) recovers
      it** — final numbers below are on par with the `@turbo` baseline.

      | grid | metric | HEAD `@turbo` | KA-only (rejected) | dual path (final) |
      |---|---|---|---|---|
      | 128² | `run!` | 312 ms | 454 ms | 305 ms |
      | 256² | `run!` | 2793 ms | 3744 ms | 2933 ms |
      | 128² | `update_second_derivatives!` | 11.7 µs | 247 µs | 15.2 µs |
      | 256² | `update_second_derivatives!` | 88.9 µs | 987 µs | 82.5 µs |

      Findings that drove the decision: KA-CPU stencils were ~10–20× slower than
      `@turbo` (~35–45% on the full FFT-dominated run); CPU threading (`-t auto`, 16
      cores) did **not** help (256² `run!` 4630 ms — launch overhead on small
      kernels); microbench (`dxx!` @ 256²) plain `@inbounds` 18.7 µs ≈ `@simd`
      18.4 µs ≪ KA 150 µs. User chose the dual path; implemented and verified: the
      "no perf loss when AD is off" requirement is met, physics unchanged
      (−267.1312 m), full suite passes. GPU path unbenchmarked (no hardware).

## 3. Phase 1 — Inversion API (core package, no AD deps) — **DONE (2026-07-07)**

New files under `src/inverse/` (`diffmode.jl`, `observables.jl`, `encodings.jl`,
`regularization.jl`, `problem.jl`); old `src/inversion.jl` kept intact (EKP ext
still uses it). Tests: `test/test_inversion_api.jl` (16/16, wired into runtests).

- [x] `AbstractInversion`; `IceLoadInversion`, `ParameterInversion` (same fields:
      sim template, encoding, observations, regularizations, diffmode; plus a
      precomputed `extract_times`). Shared `loss`/forward machinery via the
      abstract supertype. Constructors: `(sim, encoding, observations;
      regularizations = (), diffmode = TangentMode())`.
- [x] `AbstractDiffMode`; `TangentMode(; batch=8)` / `AdjointMode(; checkpoint_every=10)`
      + `requires_encoding`. Constructors validate `TangentMode ⇒ encoding !== nothing`.
- [x] Observables: `AbstractObservable` + `VerticalUpliftObservable` (u+ue),
      `VerticalUpliftRateObservable` (dudt), `RelativeSeaLevelObservable`
      ((z_ss−z_ss_ref)−(u+ue)); `HorizontalDisplacementRateObservable` intentionally
      undefined. `Observation{O}` wrapper (tag, `Vector{CartesianIndex{2}}`, times,
      flat `data` points-fastest, scalar/vector `σ`) with dim-mismatch validation.
      `observable_value(tag, sim, idx)` (scalar reads) + `extract!`.
- [x] Observation times: `forward_predict!` advances via `advance_with_output!` to
      each time in the sorted-unique `extract_times`, extracting matching obs at each
      stop. **Added `reset_state!(sim)`** (src/state.jl) so each `loss` eval restarts
      from the IC — without it, repeated runs continued from the previous end-state.
- [x] Encodings: `AbstractEncoding`; `reconstruct!` methods for `Test1Encoding`
      (3 Vialov domes + bimodal log10 viscosity; 3K+13 params) and `Test2Encoding`
      (4-Gaussian log10 viscosity + 2 densities; 19 params), Enzyme-legal
      (indexed θ reads + broadcasts). `EOFEncoding`/`AutoEncoding`/
      `VariationalAutoEncoding` are erroring stubs. `nparams` on each.
  - [x] Decision: encodings write `effective_viscosity` **directly** (`10^log10η`),
        which is what the lat-variable Maxwell path reads. Layered-viscosity →
        `get_effective_viscosity_and_scaling` mapping is deferred to real-data use.
- [x] Regularization & bounds: `L2Reg(λ)` (‖θ‖²), `SurfaceSmoothnessReg(λ)`
      (Σₖ‖∇(Hₖ+b_ref)‖² via FD stencils), `DecodedBounds(quantity, lo, hi, λ)` with
      named `BoundedQuantity` types (`Log10Viscosity`/`UpperMantleDensity`/
      `LithoDensity`, no closures) → `λ·Σ[relu(lo−d)²+relu(d−hi)²]`.
  - [x] **Generalize to one configurable Tikhonov regularizer** (decided 2026-07-08,
        implemented 2026-07-09; §1 Regularization row, §4b A). `TikhonovReg(target,
        order; λ, weights)`; `L2Reg`/`SurfaceSmoothnessReg` now thin constructors
        over it. See §4b A for the full writeup.
- [x] `loss(prob, θ)` = reconstruct! → reset_state! → forward run to obs times →
      ½Σ‖(pred−data)/σ‖² + Σ penalties. **Verified**: `loss(θ_true)=0` exactly,
      monotone increase under viscosity perturbation, realistic subsidence data.
- [x] Stubs `gradient!`/`solve!` (problem.jl) + weakdep/ext skeletons wired in
      Project.toml: `FastIsostasyEnzymeExt` (Enzyme; placeholder gradient!, Phase 2),
      `FastIsostasyCheckpointingExt` (Checkpointing+Enzyme; Phase 5),
      `FastIsostasyOptimExt` (real `solve!` via Optim L-BFGS, runnable once
      `gradient!` exists). Core resolves & loads without the weakdeps.

**Caveats surfaced for later — all three resolved 2026-07-08:**
(1) **[resolved]** `L2Reg` raw-θ domination → the generalized Tikhonov regularizer
(§1 Regularization row, §3 refactor box) makes order-0 magnitude penalties accept
per-component weights; user's steer: usually apply the **gradient** (`order=1`,
smoothness) penalty anyway.
(2) **[resolved]** Observation extraction → vector-based KA gather on linear indices,
`get_backend`-dispatched (§1 Obs-extraction row); replaces the CPU scalar indexing.
(3) **[resolved / non-issue]** The Vialov `(·)^(3/8)` infinite margin slope never
enters an AD path: `reconstruct!` writes `H` directly and the loss depends on `H`
and displacement, not on the analytic `∂H/∂r`. The only ∇H is the FD stencil inside
the surface-smoothness reg — bounded by construction, not the analytic singularity.

## 4. Phase 2 — Tangent (forward) engine + AD validity test — **go/no-go PASSED (2026-07-09)**

- [ ] `FastIsostasyEnzymeExt`:
  - [x] **Enzyme activity map for `Simulation`** (2026-07-08): written as
        `docs/src/inversion_ad_activity_map.md` — the field-by-field `Const` vs
        `Duplicated` reference (domain/constants/opts/plans/outputs/timer Const;
        `now`, `solidearth`, `prealloc` buffers, and the ice-load `H_itp.X` shadowed).
        **`apply_bc!` done (2026-07-08, §2 audit):** OffsetBC projection rewritten as
        `X .-= (dot(bc.W, X) − bc.x_border)`, `bc.buffer` field dropped → `bc` is
        `Const`, `u` is `Duplicated`, no custom rule. Full suite passes (physics
        unchanged); all 7 `precompute_bc` constructors updated.
  - [x] `EnzymeRules.inactive` for `t_computation!` (2026-07-08): one-liner in the
        ext (`EnzymeRules.inactive(::typeof(t_computation!), args...) = nothing`);
        core stays Enzyme-free. Validated in `test/test_ad_rules.jl` (a function that
        pushes to the live timer still differentiates to the exact primal derivative).
        Printing / NetCDF writers live off the RHS; left undeclared for now.
  - [x] Forward-mode `EnzymeRules` for plan application `mul!(Y, plan, X)`
        (2026-07-08): one rule on `plan::Const{<:AbstractFFTs.Plan}` (tangent = same
        transform on each shadow column; handles `Const` input → zero tangent, and
        width>1 `BatchDuplicated`). Covers complex fft/ifft **and** rfft/irfft.
        Validated vs central FD in `test/test_ad_rules.jl` (rtol 1e-5): both a
        directional derivative and the full component-wise gradient.
  - [x] `gradient!(g, prob, θ, ::TangentMode)` (2026-07-09): `Enzyme.autodiff(Forward,
        loss, Duplicated(prob, dprob), Duplicated(θ, dθ))` with a `make_zero` shadow of
        the whole `prob` (re-zeroed per direction via **`remake_zero!`** — `make_zero!`
        trips on the plans' immutable-nonzero type-params), seeding one θ component at a
        time; `set_runtime_activity(Forward)` throughout. One forward pass per θ
        component (fine for encoded low-dim θ). **`BatchDuplicated` chunking deferred**
        (loop is correct + simple; batching is a later perf optimisation). Validated in
        `test/test_ad_validity.jl` component-wise vs FD (rtol 1e-5).
- [x] **AD validity test (the go/no-go) — FULL RUN PASSES (2026-07-09).**
      `test/test_ad_validity.jl` (6/6, wired into `runtests.jl`): forward-mode Enzyme
      gradient of the **full inversion `loss`** — `reconstruct!` → fixed-step `FIEuler`
      run → data misfit, lat-variable Maxwell, 32², `SmoothTransition` — w.r.t. θ,
      `make_zero` shadow + `set_runtime_activity`, **matches central FD to rel ≈ 1e-9**
      (directional) and component-wise via `gradient!` (rtol 1e-5). `loss(θ_true) <
      1e-6` confirms the AD forward reproduces the integrator's synthetic data.
      **Two forward-path fixes to get here:**
      (a) `forward_predict!` dispatches on the algorithm — `FIEuler` uses a direct
      explicit-Euler loop (`_advance_euler!`, mathematically identical to the
      integrator) that avoids the `FIIntegrator`'s `Vector{Matrix}` stage buffers +
      deep nested type, which overflow Enzyme's static type analysis
      (`EnzymeNoTypeError` in `perform_step!`); adaptive algs keep the integrator (not
      differentiable — TangentMode v1 is fixed-step only).
      (b) `advance_with_output!` was dropped from the inversion path (its
      `_next_output_time` returns `Union{Nothing,T}` → `IllegalTypeAnalysis`); obs
      extraction now uses a precomputed `extract_plan` (θ-independent `(obs,time)` index
      map) so no `findfirst`/`Union` is on the differentiated path.
      Verified correct along the way (single-RHS bisection): custom `mul!` plan rule,
      complex `real.()`, `apply_bc!` (dot), in-place complex-buffer reuse, and routing
      θ through the shadowed sim.
      **Root cause found & fixed — `ScaledPlan` scale activation.** `plan_ifft`/
      `plan_irfft` return `AbstractFFTs.ScaledPlan`s carrying a `Float64` `scale`.
      Inside the `make_zero`'d sim Enzyme spuriously treated that scale as active and
      its shadow corrupted the tangent (1.46 instead of −0.0082); `inactive_type` and
      `set_runtime_activity` did **not** fix it (immutable `Float64` leaf activated
      by-value). **Fix:** `NormalizedPlan{P,S}` (src/convolutions.jl) wraps the raw
      unnormalized plan `P` with the exact scale `S` **as a type parameter** (compile-
      time constant → Enzyme-invisible); `normalize_plan(::ScaledPlan)` extracts
      `sp.p`/`sp.scale` (build-extract-discard) so `mul!` reproduces the `ScaledPlan`
      result **bit-for-bit** (verified max|Δ| = 0). Wired into `choose_fft_plans`
      (tools.jl) and `_plan_irfft`/`ConvolutionPlanHelpers` (convolutions.jl) + the
      CUDA ext. Full suite passes unchanged (integrators 31/31, convolutions,
      barystatic, derivatives); `test_ad_rules` 4/4.
      **`gradient!(::TangentMode)` must use `set_runtime_activity`** (for the in-place
      complex-buffer reuse) + `make_zero` shadow.
      Ext also: `inactive_type(<:AbstractFFTs.Plan)` + `inactive_type(<:NormalizedPlan)`;
      `mul!` rule narrowed to raw `cFFTWPlan`/`rFFTWPlan` (so `ScaledPlan` is traced,
      not custom-ruled — matching it tripped Enzyme's `roots_activep` assertion).
- [x] Gradient through adaptive stepping (`FIBS3`/`FITsit5`): **restricted — TangentMode
      v1 is fixed-step (`FIEuler`) only** (2026-07-09). `forward_predict!` dispatches the
      differentiable direct-Euler loop for `FIEuler` and errors/uses the (non-diff)
      integrator otherwise. Adaptive-step AD (freezing the dt-sequence) is Phase 5+.
- [x] Wire tests into `test/runtests.jl` (2026-07-08): `test/test_ad_rules.jl`
      (foundational rules) added; `test/test_ad_validity.jl` (full go/no-go) still to
      come. `Enzyme` + `FFTW` added to `test/Project.toml`.

## 4b. Pre-Phase-3 review findings (2026-07-09) — target before the docs examples

Code review of Phases 0–2 + the inversion API. Grouped by priority; B items are
silent-wrongness traps, A items are locked decisions the code doesn't yet match.

**A — roadmap/code mismatches — all three DONE (2026-07-09):**
- [x] **Obs extraction still CPU scalar loop** (`src/inverse/observables.jl`,
      `extract!`): the code was worse than described — `problem.jl` called the
      5-arg `extract!` with only 4 args (`MethodError`, confirmed by running
      `test_inversion_api.jl`: 3/16 errored). Fixed by actually wiring the
      documented design: `IceLoadInversion`/`ParameterInversion` gained a
      `linear_indices` field, computed once per observation via
      `_obs_linear_indices`/`_linear_indices` (`problem.jl`) — `obs.points`
      converted to `LinearIndices`, then `similar(field, Int, n)` + `copyto!` to
      land on the *same array family as the sim's fields* (`Matrix` on CPU,
      `CuMatrix` on GPU — no `Adapt.jl` dependency needed). `allocate_predictions`
      likewise switched from `zeros` to `similar(sim.now.u, T, n)` so `gather!`'s
      GPU branch never writes a device kernel's output into a host `Vector`.
      `_forward_run!` (both `FIEuler` and integrator branches) now passes
      `prob.linear_indices[k]` through. Verified: `test_inversion_api.jl` 16/16,
      `test_ad_validity.jl` 6/6 (extract! is on the differentiated path via
      `forward_predict!`, so this also confirms it stays Enzyme-legal). GPU
      backend correctness for `data_misfit` (comparing a possible `CuVector` pred
      against `obs.data`, always a CPU `Vector`) is unaddressed — deferred to §7
      (no CUDA hardware to validate against yet).
- [x] **`loss_and_gradient!` missing**: added the core stub (`problem.jl`,
      alongside `gradient!`) and the `TangentMode` implementation
      (`FastIsostasyEnzymeExt.jl`): same per-θ-component seeding loop as
      `gradient!`, but using `Enzyme.ForwardWithPrimal` instead of `Forward` — the
      primal (`loss(prob,θ)`, direction-independent) is read off any one pass, so
      no extra forward run is needed to also get the objective value.
      `FastIsostasyOptimExt.solve!` rewritten around `Optim.only_fg!(fg!)`
      (`fg!(F, G, θ)` computes `loss_and_gradient!` when `G !== nothing`, reuses
      its primal for `F` instead of a separate `loss` call). Verified:
      `loss_and_gradient!` output matches separate `gradient!` + `loss` exactly
      (`maxdiff_g = 0.0`, same primal) on the `test_ad_validity.jl` problem setup;
      `solve!` smoke-tested end-to-end (3 L-BFGS iterations, loss 23113 → 0.0997).
- [x] **Generalized Tikhonov regularizer** (`src/inverse/regularization.jl`
      rewritten): `TikhonovReg(target, order; λ, weights)` with `target ∈
      {ThetaTarget(idx), FieldTarget(quantity::BoundedQuantity), SurfaceTarget()}`
      and `order ∈ {Order0(), Order1()}`. `Order0` = magnitude `Σ wᵢxᵢ²` (optional
      per-component `weights`, retiring the §11 raw-θ domination issue); `Order1`
      = gradient `Σ‖∇x‖²` via the existing `dx!`/`dy!` FD stencils, valid on
      `FieldTarget`/`SurfaceTarget` only (`ThetaTarget` + `Order1` throws — no
      spatial structure on raw θ; scalar `FieldTarget` + `Order1` throws too).
      `L2Reg(λ) = TikhonovReg(ThetaTarget(), Order0(); λ)`,
      `SurfaceSmoothnessReg(λ) = TikhonovReg(SurfaceTarget(), Order1(); λ)` kept
      as thin constructors — no call-site churn, `test_inversion_api.jl`'s
      `L2Reg`/`DecodedBounds` assertions pass unchanged. `BoundedQuantity`/
      `decoded(...)` (shared with `DecodedBounds`) moved earlier in the file so
      `FieldTarget{<:BoundedQuantity}` can reference the bound. New exports:
      `TikhonovReg`, `AbstractRegTarget`/`ThetaTarget`/`FieldTarget`/
      `SurfaceTarget`, `AbstractRegOrder`/`Order0`/`Order1`. Verified with a
      standalone script exercising all target×order combinations (weighted
      `ThetaTarget` subset, `FieldTarget` order-0/order-1, the two thrown-error
      cases, `SurfaceSmoothnessReg` vs the equivalent direct `TikhonovReg` call)
      plus the full `test_inversion_api.jl` suite (16/16).

**B — correctness traps (fix before Test 1) — all six DONE (2026-07-09):**
- [x] **Vialov margin NaN under AD** (`add_vialov!`, encodings.jl): confirmed with
      a minimal Enzyme repro before touching the code — far field (`base ≪ 0`)
      differentiated to `NaN`, exactly at the margin (`base = 0`) to `Inf`. Root
      cause matches the roadmap note: `max(base,0)`'s forward-mode tangent
      correctly zeroes out on the clamped branch, but `0^(3//8)`'s own pow-rule
      derivative (`(3/8)·0^(-5/8) = Inf`) still gets *formed* before being
      multiplied by that zero tangent → `Inf·0 = NaN`. Fixed by extracting a
      `_vialov_shape(base) = (b = max(base,0); ifelse(b>0, b^(3//8), zero(b)))`
      helper: Enzyme's `ifelse` selects between the two branches' *already
      computed* tangents based on the primal predicate, so the poisoned
      `b^(3//8)` tangent is discarded (not combined arithmetically) when `b ≤ 0`.
      Re-ran the same repro through the fix: both the far-field and exactly-at-
      margin cases now differentiate to `0.0`. Also checked end-to-end through
      `reconstruct!` on a real `Test1Encoding` sim (dome far from most grid
      cells, `d/dxc` finite).
- [x] **`Test1Encoding` never validates ice snapshots**: added
      `_check_ice_snapshots(sim, encoding)` (`problem.jl`, dispatches to a no-op
      for every encoding except `Test1Encoding`), called from the shared
      `IceLoadInversion`/`ParameterInversion` constructor. Checks both
      `length(ice_snapshots(sim)) == length(enc.knot_times)` and
      `sim.bcs.ice_thickness.H_itp.t == enc.knot_times`. Verified: mismatched
      count and mismatched times each throw `ArgumentError`; a matching encoding
      still constructs.
- [x] **Duplicate times in one `Observation` corrupt the misfit silently**:
      `Observation(...)` now checks `allunique(times)`, throwing `ArgumentError`
      on a duplicate. Verified.
- [x] **Obs times outside `t_span` unvalidated**: added `_check_obs_times(sim,
      observations)` (`problem.jl`), called from the shared inversion
      constructor — throws `ArgumentError` if any observation time falls outside
      `sim.timer.t_span`. Verified both before-`t_span[1]` and after-`t_span[2]`
      cases throw; an in-range time still constructs.
- [x] **No `length(θ) == nparams(encoding)` check in `loss`**: one-liner guard
      added at the top of `loss` (skipped when `encoding === nothing`, the
      full-field/`AdjointMode` case). Verified: an oversized `θ` throws
      `DimensionMismatch` instead of silently reading only the first
      `nparams(encoding)` entries.
- [x] **TangentMode + adaptive alg = cryptic `EnzymeNoTypeError`**: added
      `_require_fieuler(prob)` in `FastIsostasyEnzymeExt.jl`, called at the top
      of both `gradient!(::TangentMode)` and `loss_and_gradient!(::TangentMode)`
      (the latter added in the A-item pass, same restriction applies). Verified:
      calling `gradient!` on a `ParameterInversion` built from a sim with the
      default adaptive `FIBS3` algorithm now errors immediately with the
      documented "TangentMode v1 is fixed-step only" message instead of
      whatever `EnzymeNoTypeError` the integrator branch would have produced.

  All six verified together via `test/runtests.jl` (unchanged pass counts:
  barystatic 2/2, convolutions 1/1, data loaders 14/14, derivatives 3/3,
  indexing 2/2, integrators 31/31, inversion API 16/16, AD rules 4/4, AD
  validity 6/6) plus a standalone script exercising each new guard directly
  (both the error and the accept path for every check).

**C — API polish (optional) — all three DONE (2026-07-09):**
- [x] **`gradient!` without the Enzyme ext is a bare `MethodError`**: the 3-arg
      dispatcher (`gradient!(g,prob,θ) = gradient!(g,prob,θ,prob.diffmode)`) now
      lives in core (`problem.jl`), always resolves; a new 4-arg fallback
      `gradient!(g,prob,θ,::AbstractDiffMode)` also lives in core and errors with
      "requires FastIsostasyEnzymeExt to be loaded". The ext keeps only its
      concrete-type overrides (`::TangentMode`, `::AdjointMode`), which win by
      dispatch specificity once loaded. Same split applied to
      `loss_and_gradient!` (added in the A-item pass — same bare-`MethodError`
      problem, same fix). Verified: calling `gradient!`/`loss_and_gradient!` in a
      session with `using FastIsostasy` but no `using Enzyme` now errors with the
      informative message instead of `MethodError`; the full ext-loaded test
      suite (including `test_ad_validity.jl`) is unaffected — the ext's
      mode-specific methods still take priority.
- [x] **`Test2Encoding{T}` dead type param**: dropped. `Test2Encoding` has no
      fields (unlike `Test1Encoding`, whose `T` is inferred from real
      `knot_times`/`radii`/`visc_amps` data), so its `T` was pure decoration, and
      the `Float32` default actively contradicted the §1 Precision row
      (`Float64` for inversion runs). Now `struct Test2Encoding <:
      AbstractEncoding{Float64} end`, no keyword constructor needed. No call-site
      changes (`Test2Encoding()` unaffected); `test_inversion_api.jl`/
      `test_ad_validity.jl` unchanged.
- [x] **`NormalizedPlan` hardcoded `Float64(scale)` vs a Float32 `ScaledPlan`
      — investigated, confirmed non-issue** (same pattern as the §3 caveat (3)
      Vialov write-up: flagged as a risk, resolved by closer analysis rather than
      a code change). `y .*= Float64(scale)` on a `ComplexF32` `y` is a *single*
      rounding step regardless of the scale's stored type: IEEE754 double
      rounding is provably safe here because Float64's mantissa (52 bits) is
      more than double Float32's (23 bits), so
      `round32(round64(x·s)) == round32(x·s)` always — no accumulation, no FMA
      chain, just one scalar multiply. Verified two ways: (1) 2M random
      Float32×Float32 pairs computed both directly and via a Float64
      intermediate, zero mismatches; (2) end-to-end, wrapping the *same*
      `AbstractFFTs.ScaledPlan` object (both `ifft` and `irfft`, sizes 5×7
      through 256×256, prime/composite/power-of-two) with `normalize_plan` and
      comparing `sp * X` vs `mul!(y, normalize_plan(sp), X)` — bit-identical
      (`maxdiff = 0.0`) in every case. (An earlier draft of this check built two
      *separate* `plan_irfft` calls with different FFTW flags — `MEASURE` vs the
      default — and saw ~1e-7 diffs; that was FFTW picking a different algorithm
      between the two plans, unrelated to the scale type, and not how
      `normalize_plan` is actually used in the codebase, which always wraps one
      already-built plan.) No code change; left as-is with this note in place of
      the roadmap concern.

**D — noted, fine to defer:** `SurfaceSmoothnessReg.penalty` allocates 3 work
arrays per loss eval; `TangentMode.batch` dead (deferred by design);
`_extract_times(())` on empty observations errors cryptically.

## 4c. Pre-Phase 3 API modifications (assessed & design locked 2026-07-09)

Do these **before** the Phase 3 docs examples (they lock the API). Order: loss
first (constructor change item 2 also touches), then `SimulatedObservable`.
API-symmetry is a guideline applied while doing both, not a separate task; the
Hessian idea moved to §11 (deferred, needs Phase 5 first).

- [x] **Pluggable loss — `AbstractLoss` stored as a problem field** (~½ day;
      done 2026-07-09). Implemented exactly per the locked design: `AbstractLoss`
      + `DefaultLoss <: AbstractLoss` (`misfit(::DefaultLoss, preds,
      observations)` = the old `data_misfit` body, now removed) in
      `src/inverse/problem.jl`. `IceLoadInversion`/`ParameterInversion` gained a
      `lossmodel` field (new type param `LM`, last field); constructor kwarg
      `lossmodel = DefaultLoss()`. `loss(prob, θ)` now calls
      `misfit(prob.lossmodel, preds, prob.observations)` instead of
      `data_misfit(prob, preds)`. Exported: `AbstractLoss`, `DefaultLoss`,
      `misfit`. **No Enzyme ext change needed** — confirmed by re-running
      `test_ad_rules.jl` (4/4) and `test_ad_validity.jl` (6/6) unchanged;
      `prob`'s `make_zero`/`remake_zero!` shadow already covers the new field.
      New test in `test_inversion_api.jl` ("pluggable loss (AbstractLoss)"):
      defines a `ScaledLoss <: AbstractLoss` outside the package and checks
      `loss` with it equals `scale * loss(DefaultLoss())` — confirms the
      extension point works end-to-end for a user-defined loss. Full suite:
      18/18 inversion API (was 16/16), all other counts unchanged.
- [x] **`SimulatedObservable` — forward-run virtual stations** (~1 day; done
      2026-07-09). Implemented the forward-side half exactly as scoped
      (inversion-side unification deferred — see below).
      `src/inverse/observables.jl`: `points_to_linear_indices(points, field)`
      extracted as a shared helper (`problem.jl`'s `_linear_indices` is now a
      one-line wrapper over it — no behavior change, `test_ad_validity.jl`
      6/6 confirms). New `mutable struct SimulatedObservable{O,T,LI,D}` (tag,
      points, times, backend-promoted linear indices, flat `data` — points-
      fastest then times, `Observation`'s convention — and cursor `k`);
      `SimulatedObservable(tag, points, times, sim)` builds it from `sim`'s
      current field layout; `attach_simobs!(sim, tag, points, times)` builds
      **and** `push!`s it onto `sim.simobs`. `record!(so, sim)` gathers via the
      existing `observable_field` + dual-path `gather!` and advances `k`.
      `Simulation` (`src/simulation.jl`) gained a `simobs::VO` field (new last
      type param `VO`; default `simobs = SimulatedObservable[]` kwarg on the
      outer constructor) — attach *after* construction (`attach_simobs!`),
      since building a `SimulatedObservable` needs a live `sim` to read the
      field layout from (avoids the circularity of an embedded-at-construction
      design). `src/integrators.jl`: `_next_simobs_time`/`next_simobs_time`
      fold simobs pending times into `_next_output_time`; `advance_with_output!`
      fires `record!` for every station matching the stop time, **after**
      `nc_affect!`/`nout_affect!` (documented order). Entirely off the
      differentiated path — confirmed by re-running `test_ad_rules.jl` (4/4)
      and `test_ad_validity.jl` (6/6) unchanged (inversions use
      `_advance_euler!`, never `advance_with_output!`).
      New `test/test_simulated_observable.jl` (9/9, wired into
      `runtests.jl`): a station's recorded values match an independent
      full-field extraction at the same time; `run!` is a no-op on `simobs`
      bookkeeping when none are attached; multiple stations with different
      tags/point counts/time grids keep independent cursors. Full suite
      unaffected otherwise (barystatic 2/2, convolutions 1/1, data loaders
      14/14, derivatives 3/3, indexing 2/2, integrators 31/31, inversion API
      18/18, AD rules 4/4, AD validity 6/6).
      **Deferred, not done:** the inversion-side refactor (building
      `SimulatedObservable`s from `Observation`s inside `IceLoadInversion`/
      `ParameterInversion` so both sides share one type). Left alone
      deliberately — `forward_predict!`'s existing `extract_plan` +
      `linear_indices` mechanism is already Enzyme-validated and routing it
      through `SimulatedObservable` would touch the differentiated path for a
      cosmetic gain only (the assessment's point: the RSL-storage concern was
      already solved there). Revisit if the Test 1/2 docs examples want a
      shared predicted-vs-observed plotting object — cheap to add a thin
      `Observation → SimulatedObservable` converter for docs/plotting use
      without changing `forward_predict!` itself.
- [ ] **API symmetry forward/inverse — guideline, not a refactor.** Already
      reasonably parallel (`Simulation`/`run!` vs `IceLoadInversion`/`solve!`).
      Close the two concrete gaps via the items above (shared observable types;
      loss/reg objects configured at construction like `SolverOptions`). A
      CommonSolve-style rename of `run!` is public-API breakage — only worth
      considering at the v2.0 boundary, default is don't.

## 5. Phase 3+4 — Synthetic inversion tests (both TangentMode)

**Docs examples (decided 2026-07-09):** each test doubles as a dedicated docs
example — Test 1 → **"Inverse ice history"** (`IceLoadInversion`), Test 2 →
**"Inverse calibration"** (`ParameterInversion`). Write them as docs pages under
`docs/src/examples/` wired into `docs/make.jl`, not just test files; the §7
`inversion_ad.md` docs item then links to them instead of duplicating.

### Test 1 — joint ice + bimodal viscosity (`test/test_inversion_vialov.jl`)

Setup (ground truth = FastIsostasy run with true θ, fixed RNG):
- Ice: superposition of 3 radially symmetric Vialov domes,
  `H(r) = H_c [1 − (r/L)^{4/3}]^{3/8}` (n = 3), fixed radii `L_i`; unknown constant
  centers `(xᵢ, yᵢ)`; unknown `H_{c,i}(t)` as piecewise-linear values at ~5 knots
  following a **glacial-cycle sawtooth**: slow growth over ~80–90 % of the span,
  rapid deglaciation at the end (knot times fixed and asymmetric to resolve the
  deglaciation; only knot *values* are unknowns).
- Viscosity: background `log₁₀η_bg` + low Gaussian anomaly (μ₁ ∈ ℝ², σ₁) + high
  Gaussian anomaly (μ₂ ∈ ℝ², σ₂); amplitudes fixed (e.g. ∓1 decade) — schematic
  East/West Antarctica dichotomy.
- θ = 3·5 (knots) + 6 (centers) + 1 + 3 + 3 = **28 parameters** via `Test1Encoding`.
- Observations: `VerticalUpliftRateObservable` at ~1 % of cells at `t_end`;
  `RelativeSeaLevelObservable` at ~1 % of cells at ~10 times.
- [x] Ground-truth generation script + stored synthetic obs (fixed seed).
- [x] Inversion from well-informed initial guess (~10–20 % perturbation),
      `TangentMode` + Optim L-BFGS (`FastIsostasyOptimExt`: `solve!(prob, LBFGS())`),
      `DecodedBounds` on viscosity range and `H ≥ 0`.
- [x] Assertions: gradient check at θ₀ vs FD; monotone loss decrease; parameter
      recovery within tolerance (define per-parameter tolerances when writing).

**DONE (2026-07-10).** Docs example `docs/src/examples/inverse_ice_history.jl`
(Literate, wired into `docs/make.jl` + `example_pages`) + lean CI test
`test/test_inversion_vialov.jl` (9/9, wired into `runtests.jl`, ~5 min —
Enzyme-compilation-dominated, not iteration-dominated, so trimming iters/dt
doesn't help; dt must stay 500 for physics). Verified end-to-end **including the
CairoMakie plots**: gradient check AD==FD to 5 figures, loss 2.2e5 → ~4, **ice
field recovered to ~40 m on a 2494 m peak (1.6 %)**, viscosity field to ~0.02
decades. (Initially validated in a clean env to sidestep a broken NLsolve ext;
that ext is now fixed — §11 — so the docs env builds directly.)

Deviations from the original spec, with rationale (all forced by what actually
works under Enzyme / what is well-posed):
1. **Observations: single `VerticalUpliftObservable` at 10 times**, NOT
   rate@t_end + RSL@10×. **Mixing observable types trips Enzyme** — the
   `observations` vector becomes abstractly typed
   (`Observation{O,…} where O<:AbstractObservable`) and the per-observation
   `observable_field(obs.tag, sim)` dispatch is dynamic → `EnzymeInternalError`
   in both `gradient!` (Forward) and `loss_and_gradient!` (ForwardWithPrimal).
   Homogeneous (same-tag) multi-observation is fine (concrete eltype). A
   10-time uplift series carries the same temporal ice-history constraint. The
   general fix (store `observations` as a Tuple + type-stable/unrolled misfit
   loop so mixed tags stay concrete) is future work — see §11.
   **RSL under AD is also still unvalidated** (Phase 2 only proved
   `VerticalUpliftObservable`); the heterogeneous failure masked it here, so it
   stays a §11 open item.
2. **`Test1Encoding` gained a `scale` field** (default all-ones → fully
   backward-compatible, `test_inversion_api` 18/18 unchanged): physical value =
   `θ[i]·scale[i]`, applied as one broadcast at the top of `reconstruct!`
   (Enzyme-legal). This lets the optimization variable θ be dimensionless/O(1)
   while the physics sees metres-of-thickness / metres-of-position / decades. It
   is the fix for the §11 normalization item **for L-BFGS conditioning**: without
   it, raw-θ gradients span ~1 (an ice knot) to ~2e5 (log10η_bg), L-BFGS stalls
   at a poor minimum AND a diagonal preconditioner overshoots
   (`10^(2e5)=Inf` → line-search assertion). With it, plain `solve!(prob, θ0)`
   with default `LBFGS()` converges cleanly. §11 normalization item ticked.
3. **Well-separated, non-overlapping domes** (radii 0.7–0.8e6, centres ±1.3e6 on
   a ±3e6 domain), NOT the originally-vague overlapping layout. Overlapping domes
   (radii ≈ separations) are genuinely ill-posed: GIA spatially low-passes the
   load, so overlapping-dome centres trade off and the ice *field* error hit
   ~1000 m even at low loss. Separated domes make the load identifiable (field
   error ~40 m). Viscosity-anomaly *locations* still trade off (~30 km) but the
   viscosity *field* is recovered (~0.02 decades) — documented as the honest
   expected behaviour, asserted on fields not individual centres.
4. `DecodedBounds` mentioned in the spec are **not needed** — normalized θ +
   the informed initial guess keep the run in the physical region without them;
   left out to keep the example minimal (could add as a showcase later).

### Test 2 — viscosity (4 Gaussians) + densities (`test/test_inversion_viscdens.jl`)

- Ice: **known** (true sawtooth Vialov forcing from Test 1).
- Unknowns via `Test2Encoding`: background `log₁₀η_bg`; 4 Gaussians with centers,
  widths **and amplitudes** (4·4 = 16); `rho_uppermantle`, `rho_litho`
  → **~19 parameters**.
- Observation: `VerticalUpliftObservable` as full (x, y, t) field at output times.
- [x] Ground truth + inversion + same assertion pattern as Test 1.
- [x] Check identifiability of densities vs viscosity (expect correlated params;
      document, don't over-tune).

**DONE (2026-07-10).** Docs example `docs/src/examples/inverse_calibration.jl`
(Literate, wired into `docs/make.jl` + `example_pages`) + lean CI test
`test/test_inversion_viscdens.jl` (10/10, wired into `runtests.jl`, ~4.5 min).
Verified end-to-end **including the CairoMakie plots**: gradient check AD==FD to
~9 figures, **loss 5.7e5 → ~1e-2**, viscosity field recovered to ~1e-4 decades,
and **both densities recovered to <1 kg/m³** (ρ_um 3400→3400.3, ρ_litho
3200→3198.9).

Setup notes / deviations:
1. **`Test2Encoding` gained the same `scale` field as `Test1Encoding`** (§11
   normalization; default all-ones → backward-compatible, `test_inversion_api`
   18/18 and `test_ad_validity` 6/6 unchanged): physical = `θ·scale`, one broadcast
   at the top of `reconstruct!`. Required for the same reason as Test 1 — raw-θ
   gradients here span ~5e4 (a density) to ~1e7 (log10η_bg).
2. **Known ice = a single broad central Vialov dome** (radius 2000 km, sawtooth in
   time), NOT the 3 separated Test-1 domes. A broad load puts *all four* viscosity
   anomalies under ice so they are sensed; separated corner domes would leave the
   anomaly under the ice-free corner unconstrained. Built cleanly via
   `TimeInterpolatedIceThickness(knot_times, H_snapshots, domain)` from an inline
   `vialov_dome` helper (no internal `add_vialov!` needed); `Test2Encoding`'s
   `reconstruct!` never touches the ice, so it stays fixed across the inversion.
3. **Densities turned out well-identified, contra the roadmap's caution.** The
   full-field × 5-times observation is 3380 constraints on 19 params — rich enough
   to break the viscosity/density degeneracy (both densities to <1 kg/m³). The
   example documents this honestly: the correlation is real but the rich `(x,y,t)`
   data resolves it; sparse data would reintroduce it (suggest fixing densities or
   adding a prior then). Test asserts densities to <50 kg/m³ (loose, robust) and
   the viscosity field to <0.05 decades.
4. Single observable type (`VerticalUpliftObservable`) — no heterogeneous-obs
   Enzyme issue (§11); full-field means all interior cells (26²) as `points`.

**After Test 2**: user provides physical-coordinate observations → implement
coordinate-based `Observation` with differentiable bilinear interpolation (§11).

## 6. Phase 5 — Adjoint (reverse) engine + checkpointing

**Started 2026-07-10. Both foundational items below are DONE** (reverse-mode plan
rules — complex + rfft/irfft — and the in-memory snapshot machinery). Next:
forward recording → checkpointing ext → `gradient!(::AdjointMode)` → Test 3.

- [x] Reverse-mode `EnzymeRules` for plan `mul!` (adjoint = scaled inverse
      transform; complex + rfft/irfft). **DONE (2026-07-10).** One shared
      `EnzymeRules.augmented_primal` on `mul!(Y, plan::_RawPlan, X)` (just runs the
      transform — Const linear operator, no tape) + **three** `EnzymeRules.reverse`
      methods dispatched by plan kind:
      • **complex `cFFTWPlan`** (`fft`/`bfft`, the `update_dudt!` path): each raw DFT
        operator is *complex-symmetric* (`Wᵀ=W`), so `Pᴴ = conj(P)` and
        `Pᴴ·Ȳ = conj(P·conj(Ȳ))` — the **same** plan, no complementary transform.
        Serves both the forward `W` and the raw inverse `W̄` inside a `NormalizedPlan`
        (whose `y .*= scale` is differentiated natively).
      • **real `rfft` (`rFFTWPlan{Float64}`)**: `R = S₁·F` projects dim-1 to `m=N÷2+1`
        rows, so `Rᴴ(Ȳ) = Re(bfft(zeropadₙ(Ȳ)))` (derived from the full complex DFT —
        no scaling ambiguity, `Re` handles DC/Nyquist).
      • **real `brfft` (`rFFTWPlan{<:Complex}`, raw plan inside the irfft
        `NormalizedPlan`)**: `Bᴴ(Z̄) = D ⊙ rfft(Z̄)`, `D` **doubling** the interior
        dim-1 rows (DC and, for even `N`, Nyquist stay 1 — the transpose of `brfft`'s
        hermitian doubling).
      The rfft/brfft adjoints need the *complementary* transform (`bfft`/`rfft`),
      built on the fly via `AbstractFFTs` (correctness first; a threaded plan is a
      later perf optimisation). `Y` is fully overwritten ⇒ its cotangent is zeroed in
      `reverse` (distinct dest/src buffers guaranteed). Validated in
      `test/test_ad_rules.jl` (now **8/8**): reverse gradient vs central FD (rtol 1e-5)
      for the complex `fft→normalized-ifft` path (also vs forward-AD, rtol 1e-8) and
      the `rfft→normalized-irfft` roundtrip at **both even (n=6) and odd (n=7)** sizes
      (Nyquist present only for even). Real transforms reuse buffers ⇒
      `set_runtime_activity(Reverse)` needed (as in `gradient!`). CUFFT plans are
      `AbstractFFTs.Plan`s too ⇒ same rules on GPU (Phase 6 verify).
- [x] **State-snapshot machinery** (shared with the JLD2 restart roadmap)
      **DONE (2026-07-10, in-memory backend).** `src/snapshot.jl`: `StateSnapshot(sim)`
      allocates a buffer (deepcopy of `sim.now` + BSL + clock); `snapshot!(buf, sim)` /
      `restore!(sim, buf)` copy in place (reused buffers → allocate once). Captures
      **all** mutated state: every `CurrentState` array + the nested `ColumnAnomalies`,
      the scalars (`count_sparse_updates`, `z_bsl`, `V_af/V_pov/V_den`, `delta_V`),
      the BSL (`z`/`A`/`residual`, recursing through `CombinedBSL`), and `timer.t`.
      Timer logging vectors (`t_vec`, `t_computation`) intentionally skipped
      (instrumentation, AD-inactive, don't affect the trajectory). Exported. Validated
      in `test/test_snapshot.jl` (11/11): snapshot→run→restore→re-run reproduces
      `u`/`ue`/`z_ss`/`count_sparse_updates`/`bsl.z` **bit-for-bit**, and a mid-run
      snapshot round-trips exactly. **On-disk (JLD2) backend** deferred to the restart
      roadmap. See [[project_restart_roadmap]].
- [ ] Forward recording: accepted `(tₖ, dtₖ)` sequence per save interval + snapshot
      at each interval boundary.
- [ ] `FastIsostasyCheckpointingExt`: periodic schedule over save intervals
      (known count); inside one interval, recompute forward storing every accepted
      step, then Enzyme-reverse each `advance_step!` with frozen dt, threading the
      shadow state and accumulating `∂θ` (or `∂field` for full-field inversions).
- [ ] `gradient!(g, prob, θ, ::AdjointMode)`; validate vs TangentMode gradients on
      Test 2's setup (same θ, same loss → same gradient).
- [ ] **Test 3 (full-field)**: Test 2 setup but invert the full 2D
      `effective_viscosity` field (no encoding) with `L2Reg`; AdjointMode + L-BFGS;
      assert recovery of the 4-Gaussian pattern up to regularization bias.
- [ ] `IceLoadInversion` end-to-end: encoded thickness `H(x, t)` + surface-smoothness
      reg over a short glacial-cycle toy problem.

## 7. Phase 6 — GPU, deferred paths, cleanup

- [ ] Enzyme through KA kernels on CUDA: rerun the Phase-2 validity test on GPU;
      wire into `test/runtests_gpu.jl`.
- [ ] CUFFT plan rules (should be covered by AbstractFFTs-typed rules; verify).
- [ ] **Semi-implicit path refactor + AD** (Appendix A): step-kind trait,
      `semi_implicit_step!` map, then Enzyme through it (linear in `u` → cheap
      exact adjoint). Also fixes the latent `ks[1]` hazard described there.
- [ ] Port AD to `RealMaxwellMantle` (rfft/irfft rules exist by then).
- [ ] Float32 gradient-quality study (vs Float64) on Test 2; document guidance.
- [ ] Deprecate old `inversion.jl` API + EKP ext or rebase EKP as another engine.
- [ ] Docs page `docs/src/inversion_ad.md`: API, examples from Tests 1–2, caveats
      (nonsmoothness, ε choice, precision, shelf insensitivity of the load).

## 8. Final fix — pre-existing semi-implicit NaN (found 2026-07-07)

- [ ] **Fix the `NaN` in the semi-implicit MaxwellMantle path.** Laterally-constant
      / rigid lithosphere + `FIEuler` fixed step (the documented Bueler implicit
      benchmark) returns `NaN`. Confirmed pre-existing and independent of the AD work:
      reproduces on a clean `HEAD` worktree and on the untouched `RealMaxwellMantle`
      code; the explicit laterally-variable path is unaffected. Candidate causes:
      `FreqDomainViscosityLumping` producing a bad `R`/`effective_viscosity`, Float32
      overflow in the CN update at high wavenumber, or a `dt_min` /
      `dt_sparse_diagnostics` interaction. Must be fixed before the Phase-6
      semi-implicit AD refactor (§ Appendix A) has anything valid to differentiate.

---

## 9. Appendix A — Semi-implicit MaxwellMantle refactor (DEFERRED, Phase 6)

`update_dudt!(dudt, u, sim, t, ::MaxwellMantle, ::AbstractLithosphere)`
(src/deformation.jl) is a Crank–Nicolson spectral update masquerading as an RHS:
it reads `dt` from `sim.opts.diffeq.dt_min`, computes
`û₊ = ((∇ − dt/2·β)·û + dt·F̂) / (∇ + dt/2·β)`, writes into `u` and `sim.now.u`,
and never writes `dudt`. Consequences: only correct when called exactly once per
fixed step; wrong inside multi-stage RK; **latent hazard**: `ks[1]` in `init_fi`
is primed from this RHS and stays uninitialized (`similar`) memory on this path.

Target design (when this phase starts):
1. `stepkind(mantle, litho)` trait; `SemiImplicitStep()` for this combination.
2. Physics as a map: `semi_implicit_step!(unew, u, sim, t, dt)` — pure w.r.t. `u`,
   `dt` explicit, diagnostics chain called inside for consistency at `t`.
3. Driver: fixed-step map loop; state bookkeeping (`sim.now.u .= u`) in the driver,
   not in physics; error out if combined with an adaptive algorithm.
4. AD unit stays uniformly `advance_step!(unew, u, sim, t, dt)` for both step
   kinds. The CN update is linear in `u` → exact, cheap, stable adjoint.

## 10. Appendix B — Smoothing spec

- Clamp: `smooth_relu(x; ε) = (x + √(x² + ε²)) / 2`. Branch-free (GPU-friendly),
  symmetric bias, error ≤ ε/2 confined to |x| ≲ ε, → `max(0, x)` as ε → 0.
  Preferred over softplus (one-sided bias everywhere, overflow guards needed).
- Step: `smooth_heaviside(x; ε) = (1 + x / √(x² + ε²)) / 2` for grounded/ocean
  masks; same ε.
- ε expressed in meters of height-above-flotation; start with 1–10 m; verify
  forward-model deviation vs sharp run is below observational noise.
- The same `smooth_relu` powers ice-thickness positivity and the `DecodedBounds`
  hinge penalties (`smooth_relu(lo − d)² + smooth_relu(d − hi)²`).
- `SharpTransition` remains the default → zero cost / zero behavior change for
  forward-only users (requirement: no perf loss when AD is off).

## 11. Open questions / future work (decide when reached)

- [x] **Parameter normalization / `L2Reg` scaling** (raised 2026-07-07; reg part
      resolved 2026-07-08, conditioning part resolved 2026-07-10). Two aspects:
      (a) *regularizer* behaviour — addressed by the generalized Tikhonov regularizer
      (§1 Regularization row, §3 refactor box): order-0 magnitude penalties take
      per-component weights, common case is the order-1 gradient/smoothness penalty.
      (b) *L-BFGS conditioning* — addressed 2026-07-10 by the `Test1Encoding` `scale`
      field (§5 Test 1, deviation 2): θ optimized in dimensionless O(1) units,
      physical = `θ·scale`. Turned out to be **required**, not optional: without it
      Test 1's raw-θ gradients span ~1…2e5 and L-BFGS stalls. The same pattern
      (a `scale` broadcast at the top of `reconstruct!`) is now also in
      `Test2Encoding` (2026-07-10, Test 2) and should go into any future encoding
      used with a gradient optimizer.
- [ ] **Mixed observable types in one inversion trip Enzyme** (found 2026-07-10,
      Test 1). Passing `Observation`s of *different* tags makes `prob.observations`
      abstractly typed (`Observation{O,…} where O`), so `observable_field(obs.tag,
      sim)` dispatches dynamically inside the differentiated `forward_predict!` /
      `data_misfit` → `EnzymeInternalError` (both Forward and ForwardWithPrimal).
      Homogeneous (same-tag) multi-observation is fine. **Fix:** store `observations`
      as a `Tuple` and iterate the misfit/extract loops in a type-stable, unrolled way
      (recursion or `map` over the tuple, not `for (k,obs) in enumerate(vector)`),
      keeping each `obs` concretely typed. Until then: one observable type per
      inversion. This also blocks re-testing **RSL under AD** (Phase 2 only validated
      `VerticalUpliftObservable`; RSL's own AD-legality is still unverified because the
      heterogeneous mix failed first).
- [x] **`FastIsostasyNLsolveExt` precompile failure on Julia 1.12** (found & fixed
      2026-07-10, unrelated to AD work). Was broken three ways: no `module … end`
      wrapper (top-level `using NLsolve` → precompile error); it defined
      `update_ocean!`, which is **never dispatched** — the BSL hook is
      `update_bsl!(bsl, delta_V, t)` (`internal_update_bsl!`, sealevel.jl), so
      `PiecewiseLinearOceanSurfaceBSL` could never actually update; and it referenced a
      nonexistent `OceanSurfaceChange` type, a non-callable `A_itp(z)` (core uses
      `interpolate(z, A_itp)`), and undefined `z`/`A` in its constructor. Rewrote it as
      a proper module: keyword constructor `PiecewiseLinearOceanSurfaceBSL(; ref,
      mcp_opts)` (initialises `z`/`A` from `ref`, `residual = typemax`), a
      `surfacechange_residual` = `(z_new−z_cur)·mean(A(z_cur),A(z_new)) − delta_V`, and
      `update_bsl!(::PiecewiseLinearOceanSurfaceBSL, delta_V, t)` doing the
      box-constrained `mcpsolve` (sign-of-`delta_V` bracketing) with a
      piecewise-constant fallback when the volume residual exceeds 10 μm SLE. Also
      fixed a **dangling export**: core exported `PiecewiseLinearBSL` (undefined) —
      renamed to the actual `PiecewiseLinearOceanSurfaceBSL`. Verified: precompiles in a
      FastIsostasy+NLsolve env; a smoke test constructs it, raises BSL for `+delta_V`
      with the flooded volume closing to 1e-5, and round-trips back on `−delta_V`.
      **Docs env now precompiles all extensions** (NLsolve, Optim, Enzyme, CUDA,
      Checkpointing, Makie) — the `inverse_ice_history.md` docs-build blocker is
      cleared.
- [x] **GPU observation gather** (resolved 2026-07-08): vector-based KA gather on
      linear indices, `get_backend`-dispatched (§1 Obs-extraction row).
- [x] **CPU stencils: dual path chosen** (resolved 2026-07-07). Perf gate showed
      KA-CPU ~10–20× slower than `@turbo`; user chose plain `@inbounds` CPU loops +
      KA on GPU (option a). Implemented in `src/derivatives.jl`; perf recovered to
      the `@turbo` baseline. See the §2 perf-gate box.
- [ ] **Pre-existing NaN in the semi-implicit MaxwellMantle path** (found 2026-07-07,
      independent of the AD work): laterally-constant / rigid lithosphere + `FIEuler`
      fixed step (the documented Bueler implicit benchmark) returns `NaN`. Reproduces
      on a clean `HEAD` worktree and on the untouched `RealMaxwellMantle` code, so it
      predates this branch's AD refactor. The explicit laterally-variable path is
      unaffected. Diagnose separately (candidates: `FreqDomainViscosityLumping`
      producing a bad `R`/`effective_viscosity`, Float32 overflow in the CN update at
      high wavenumber, or the `dt_min`/`dt_sparse_diagnostics` interaction). Must be
      fixed before the Phase-6 semi-implicit AD port has anything valid to test.

- [ ] **Hessian via forward-over-reverse** (moved from §4c, 2026-07-09; deferred —
      needs Phase 5 reverse mode first). Motivating use case is **UQ**
      (Laplace/posterior covariance at the optimum) more than optimization
      (L-BFGS is already quasi-Newton; θ ≈ 20–30). Forward-over-reverse through
      a checkpointed time-stepped PDE is the hardest Enzyme configuration;
      cheap substitutes at low nθ: FD of `gradient!` (nθ extra gradient calls,
      fine as a one-off at the optimum) or forward-over-forward. No API work
      needed now — only requirement is `loss` staying pure in `(prob, θ)`,
      which it is; HVP is the primitive to build first if/when this starts.
- [ ] Physical-coordinate observations with differentiable bilinear interpolation —
      **after Test 2**; user will provide real coordinate observations then.
- [ ] `HorizontalDisplacementRateObservable`: intentionally undefined for now;
      needs a rate definition on top of `thinplate_horizontal_displacement`.
- [ ] Encoded H_c(t) knot count/placement for *real* glacial-cycle inversions
      (tests use ~5 fixed asymmetric knots).
- [ ] Whether the surface-smoothness reg uses the evolving modeled bed or a fixed
      reference bed (start with fixed reference: cheaper, state-independent reg).
- [ ] EOF/AE/VAE encodings: training pipeline lives outside FastIsostasy; only
      `reconstruct!` (decoder application) must be Enzyme-legal in-core.
