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
      (scalar write) is benign/inactive. → recorded for Phase 2 activity map.
- [x] **RHS `u`-mutation audit** (2026-07-07): `update_diagnostics!` calls
      `apply_bc!(u, sim.bcs.viscous_displacement)` on its **input** `u`
      (simulation.jl:288). For the default `OffsetBC` this is a weighted-global-mean
      removal: `bc.buffer .= bc.W.*X; X .-= (sum(bc.buffer) − bc.x_border)` — it
      mutates both the input `u` and `bc.buffer`. In the RK path each stage rebuilds
      `utmp` from scratch, so cross-stage corruption doesn't occur, but the FSAL
      `ks[1]` is computed on `integ.u`, so the projection persists into the state
      (probably intentional: keeps `u` on the zero-mean manifold). Enzyme can
      differentiate input mutation, but "input is also output" complicates the
      activity map. **Decision deferred to Phase 2**: either (a) mark the reduction
      an in-place linear op Enzyme handles, or (b) move the projection into the
      driver (once per accepted step). `bc.buffer` mutation also means the BC object
      is not `Const` — must be shadowed or the buffer thread-local.
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
- [x] `loss(prob, θ)` = reconstruct! → reset_state! → forward run to obs times →
      ½Σ‖(pred−data)/σ‖² + Σ penalties. **Verified**: `loss(θ_true)=0` exactly,
      monotone increase under viscosity perturbation, realistic subsidence data.
- [x] Stubs `gradient!`/`solve!` (problem.jl) + weakdep/ext skeletons wired in
      Project.toml: `FastIsostasyEnzymeExt` (Enzyme; placeholder gradient!, Phase 2),
      `FastIsostasyCheckpointingExt` (Checkpointing+Enzyme; Phase 5),
      `FastIsostasyOptimExt` (real `solve!` via Optim L-BFGS, runnable once
      `gradient!` exists). Core resolves & loads without the weakdeps.

**Caveats surfaced for later:** (1) `L2Reg` on raw θ is dominated by the largest-
magnitude components (Gaussian centres ~10⁶ m) and in Float32 can hide the misfit —
encoded θ should be normalized, or `L2Reg` given per-component scales (→ §11).
(2) Observation extraction uses scalar indexing (CPU); a GPU gather kernel is a
later item. (3) The Vialov `(·)^(3/8)` has an infinite margin slope — steep centre
gradients (noted in code).

## 4. Phase 2 — Tangent (forward) engine + AD validity test ⟵ go/no-go

- [ ] `FastIsostasyEnzymeExt`:
  - [ ] **Enzyme activity map for `Simulation`**: document which fields are
        `Const` (domain, FFT/conv plans, tableau, opts, output configs) vs
        `Duplicated`/shadowed (now, prealloc buffers, solidearth parameter fields,
        load/BC data). This is the reference for every autodiff call.
  - [ ] `EnzymeRules.inactive` for timers, printing, NetCDF writers.
  - [ ] Forward-mode `EnzymeRules` for plan application `mul!(y, plan, x)`:
        complex fft/ifft (explicit path) **and** rfft/irfft (ConvolutionPlan —
        already on the differentiated path via `update_deformation_rhs!` smoothing
        and `update_elasticresponse!`). Tangent rule = same transform on tangents.
        Test each rule in isolation against finite differences.
  - [ ] `gradient!(g, prob, θ, ::TangentMode)` via `Enzyme.autodiff(Forward, ...)`
        with `BatchDuplicated` shadows of the full `Simulation` (θ seeded through
        `reconstruct!`), chunked over θ.
- [ ] **AD validity test (the go/no-go)**: 32×32 grid, explicit
      `MaxwellMantle` + `LaterallyVariableLithosphere`, fixed-step `FIEuler`, few
      steps, `SmoothTransition`; Enzyme-forward gradient of a toy loss w.r.t. a
      ~5-dim θ vs central finite differences, rtol ~1e-5 (Float64). Start with a
      *single* `advance_step!`/RHS evaluation, then the full short run.
- [ ] Gradient through adaptive stepping (`FIBS3`/`FITsit5`): validate, or restrict
      TangentMode v1 to fixed-step and record the restriction here.
- [ ] Wire tests into `test/runtests.jl` (new `test/test_ad_validity.jl`).

## 5. Phase 3+4 — Synthetic inversion tests (both TangentMode)

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
- [ ] Ground-truth generation script + stored synthetic obs (fixed seed).
- [ ] Inversion from well-informed initial guess (~10–20 % perturbation),
      `TangentMode` + Optim L-BFGS (`FastIsostasyOptimExt`: `solve!(prob, LBFGS())`),
      `DecodedBounds` on viscosity range and `H ≥ 0`.
- [ ] Assertions: gradient check at θ₀ vs FD; monotone loss decrease; parameter
      recovery within tolerance (define per-parameter tolerances when writing).

### Test 2 — viscosity (4 Gaussians) + densities (`test/test_inversion_viscdens.jl`)

- Ice: **known** (true sawtooth Vialov forcing from Test 1).
- Unknowns via `Test2Encoding`: background `log₁₀η_bg`; 4 Gaussians with centers,
  widths **and amplitudes** (4·4 = 16); `rho_uppermantle`, `rho_litho`
  → **~19 parameters**.
- Observation: `VerticalUpliftObservable` as full (x, y, t) field at output times.
- [ ] Ground truth + inversion + same assertion pattern as Test 1.
- [ ] Check identifiability of densities vs viscosity (expect correlated params;
      document, don't over-tune).

**After Test 2**: user provides physical-coordinate observations → implement
coordinate-based `Observation` with differentiable bilinear interpolation (§11).

## 6. Phase 5 — Adjoint (reverse) engine + checkpointing

- [ ] Reverse-mode `EnzymeRules` for plan `mul!` (adjoint = scaled inverse
      transform; complex first, rfft/irfft for the convolution plans — mind the
      hermitian-symmetry scaling).
- [ ] **State-snapshot machinery** (shared with the JLD2 restart roadmap — write
      once, in-memory + on-disk backends). The snapshot must capture *all* mutated
      state, including scalars: `CurrentState.count_sparse_updates`, `z_bsl`,
      timer/BSL state — not just arrays. Define `snapshot!(buf, sim)` /
      `restore!(sim, buf)` and test round-trip bit-equality of a forward run.
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

- [ ] **Parameter normalization / `L2Reg` scaling** (raised 2026-07-07, Phase 1).
      `L2Reg(λ)=λ‖θ‖²` on raw θ is dominated by the largest-magnitude components
      (Gaussian centres in metres ~10⁶), and in Float32 that penalty (~10⁷) can swamp
      the data misfit past the precision floor. Fix before the tests rely on
      regularized θ: either work in a normalized θ-space (encodings map [-1,1]-ish
      latents to physical units) or give `L2Reg` per-component weights. Normalized θ
      also helps L-BFGS conditioning. Ties into the encoded-θ bounds interface below.
- [ ] **GPU observation gather** (Phase 1 left CPU-only). `observable_value` uses
      scalar indexing; a KA gather kernel is needed for `CuArray` sampling.
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
