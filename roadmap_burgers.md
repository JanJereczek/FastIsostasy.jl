# Roadmap: Burgers rheology (Ivins & Caron 2021) in FastIsostasy

Status: **design agreed, implementation not started.**
This file is the single source of truth for the Burgers-rheology effort. Tick boxes
as work lands; append decisions to §1 when they are made. Sessions should resume by
reading §1 (decisions) + §2 (physics) and finding the first unticked box.

Scope for now: **laterally constant solid-Earth parameters** (constant in x, y).
Laterally variable Burgers is out of scope until the constant case is validated (§8).

---

## 1. Locked design decisions

| Topic | Decision |
|---|---|
| Rheology | Burgers body: Maxwell element (η₁, μ₁) in **series** with Kelvin–Voigt element(s) (μ₂, η₂). Extended Burgers (I&C 2021 / Faul–Jackson continuous relaxation spectrum) approximated by **N discrete Kelvin branches (Prony series)**. Start with **N = 1** (classic Burgers); design state/params so N is a parameter. |
| Formulation | **Time-domain internal-variable formulation** in Fourier space (Bueler-style collocation), NOT the numerical Laplace–Hankel inversion used by I&C 2021. Their inversion only works for prescribed load histories (Heaviside/ramp); we need arbitrary, simulation-driven loads. |
| State split | Total viscous displacement `u = u_M + Σⱼ u_K[j]`. All branches carry the same stress (series connection ⇒ compliances add). `u_K[j]` are new persistent state fields alongside `sim.now.u`. |
| Elastic response | **Unchanged.** Farrell Green's-function convolution keeps handling the unrelaxed elastic part; the Kelvin branch(es) add only the transient anelastic band. Lithosphere term `D k⁴` unchanged inside β. |
| Time stepping | Semi-implicit Crank–Nicolson on the coupled (N+1)-field system per Fourier mode, **closed-form 2×2 solve for N = 1**. Explicit stepping rejected: Kelvin times τ₂ = η₂/μ₂ can be decades ≪ Maxwell times ⇒ stiffness. Mirror the in-place pattern of the laterally-constant `MaxwellMantle` path (sets `u` directly, uses `dt_min`), see `src/deformation.jl:52-91`. |
| Entry point | `struct BurgersMantle <: AbstractMantle` — stub already exists at `src/solidearth.jl:99`. New `update_dudt!(dudt, u, sim, t, ::BurgersMantle, litho)` methods in `src/deformation.jl`. |
| Parameters | Per Kelvin branch: relaxation strength `Δⱼ = μ₁/μ₂ⱼ` and Kelvin time `τⱼ = η₂ⱼ/μ₂ⱼ` (equivalently μ₂ⱼ, η₂ⱼ). Defaults from I&C 2021 calibrations (Faul–Jackson type: Δ ~ O(1), τ from years to centuries). |
| Compressibility | Same level of approximation as v1: apply the existing `apply_compressibility!` scaling (`src/material.jl:44-46`) to η₁. The EBM's frequency-dependent ν̃(s) is NOT implemented; documented as a caveat (§7). |
| Geometry | Flat half-space (same tradeoff FastIsostasy already makes vs. I&C's spherical compressible setting). |
| Maxwell limit | Δ → 0 (equivalently `u_K ≡ 0`) must reproduce `MaxwellMantle` to machine precision — kept as a permanent regression test. |

---

## 2. Physics (source of truth for the math)

### 2.1 Where it comes from

The current Maxwell step solves, per Fourier mode with wavenumber k = |𝐤|:

    2 η k du/dt = F − β u,        β = ρ g + D k⁴

(`nabla = 2ηk`, `beta = β` in `src/deformation.jl:52-91`). The structure behind
this is the viscoelastic half-space transfer function

    û(s) = F̂(s) / (β + 2k μ̃(s))

with μ̃(s) the Laplace-domain shear modulus. Newtonian μ̃(s) = ηs gives the ODE
above. I&C 2021 plug the extended-Burgers μ̃_EBM(s) into Wolf (1985) half-space
equations — same substitution, but they evaluate by numerical Laplace inversion.

### 2.2 EBM creep function (I&C 2021)

Dimensionless shear creep compliance:

    Ψ'(t') = 1 + t' + Δ ∫[τ_L, τ_H] F(τ') (1 − e^(−t'/τ')) dτ'

i.e. unrelaxed elastic (1) + steady Maxwell viscous (t') + transient anelastic
band (integral). The integral is a continuous spectrum of Kelvin elements ⇒
discretize as a Prony series of N branches (Δⱼ, τⱼ). N ≈ 3–5 log-spaced over
[τ_L, τ_H] usually fits to within a few percent; report N and fit error.

### 2.3 Per-mode ODE system (what we implement)

Series connection ⇒ same stress σ̂ = (F̂ − β û)/(2k) through every branch,
displacements add: û = û_M + Σⱼ û_K[j].

    2 η₁ k  dû_M/dt    = F̂ − β (û_M + Σⱼ û_K[j])                       (Maxwell dashpot)
    2 η₂ⱼ k dû_K[j]/dt = F̂ − β (û_M + Σⱼ û_K[j]) − 2 μ₂ⱼ k û_K[j]      (Kelvin branch j)

Setting û_K ≡ 0 recovers the current Maxwell equation exactly. All coefficients
depend only on k (parameters constant in x, y) ⇒ plain Fourier collocation, no
proof gap like the laterally-variable v1 case.

### 2.4 Analytic solution (for tests)

For a Heaviside disc load, each mode relaxes as a sum of N+1 exponentials whose
rates are eigenvalues of the small per-mode matrix (partial fractions of the
transfer function). For N = 1 the two rates are roots of a quadratic in s from

    β + 2k μ̃(s) = 0,   1/μ̃(s) = 1/(η₁ s) + 1/(μ₂ + η₂ s)

⇒ û(t) = û_eq (1 − A e^(−t/τ_a) − B e^(−t/τ_b)). Closed form in k; superpose
modes exactly like the existing Bueler-style analytic tests.

---

## 3. Phase 1 — Types, parameters, state

- [ ] **`BurgersMantle` fields**: replace the empty stub (`src/solidearth.jl:99`)
      with parameters for N Kelvin branches (Δⱼ, τⱼ or μ₂ⱼ, η₂ⱼ), plus docstring
      (currently says "Not implemented yet!"). Keep a convenience constructor for
      the classic N = 1 Burgers body. Defaults from I&C 2021.
- [ ] **Wire parameters through `SolidEarth`**: decide whether μ₂/η₂ live on
      `BurgersMantle` itself or as fields of `SolidEarth` next to
      `effective_viscosity` (they are scalars for now — laterally constant).
      Check how `tau` (RelaxedMantle) and `effective_viscosity` are stored for
      precedent.
- [ ] **State fields `u_K`**: add per-branch transient displacement arrays to the
      state (`src/state.jl`). Must (a) persist across steps, (b) follow `u`'s
      array type (GPU-compatible), (c) be zero-initialized, (d) be included in the
      planned JLD2 checkpoint/restart (see restart roadmap). For N branches use a
      vector of matrices or a 3D array — pick whichever plays nicer with GPU + AD.
- [ ] **Preallocation**: extra Fourier-space buffers for the 2×2 per-mode solve in
      `src/tools.jl` prealloc (audit which existing buffers are free at that point
      in the step before adding new ones).

## 4. Phase 2 — Solver

- [ ] **`update_dudt!(..., ::BurgersMantle, litho)` for laterally constant
      lithosphere**: CN discretization of §2.3. Per mode, unknowns
      (û_M, û_K)ⁿ⁺¹ solve a 2×2 linear system (N = 1) with closed-form inverse;
      broadcast over the spectrum. Mirror the `MaxwellMantle` semi-implicit
      pattern (`src/deformation.jl:52-91`): stage real fields, `mul!` with
      out-of-place plans, update `u` in place, `apply_bc!`.
- [ ] **Guard rails**: error methods for unsupported combinations
      (`BurgersMantle` + `LaterallyVariableLithosphere` ⇒ clear `error(...)` for
      now, like `RelaxedMantle` does at `src/deformation.jl:47-50`).
- [ ] **dt handling**: the laterally-constant Maxwell path uses
      `sim.opts.diffeq.dt_min` as its fixed step. Confirm this is acceptable for
      the shortest Kelvin time (CN is A-stable, but accuracy near τ₂ matters);
      document the recommended dt vs. min(τⱼ).
- [ ] **N > 1 generalization**: loop over branches; per-mode (N+1)×(N+1) solve.
      Either small dense solve per mode or a Schur-complement closed form (the
      matrix is arrowhead-structured: all branches couple only through û). Only
      after N = 1 is validated.

## 5. Phase 3 — Validation

- [ ] **Maxwell-limit regression test**: Δ → 0 (or τ₂ → 0 with μ₂ → ∞)
      reproduces `MaxwellMantle` output to near machine precision. Permanent test.
- [ ] **Analytic disc-load solution**: implement §2.4 in
      `src/analytic_solutions.jl` (two-exponential per-mode solution, N = 1);
      convergence test analogous to existing tests 1–2 in `test/`.
- [ ] **I&C 2021 qualitative reproduction**: Heaviside subsidence curves —
      enhanced early transient converging to the Maxwell curve at long times.
      Script under `examples/` or a plot-producing test.
- [ ] **GPU run**: verify the Burgers path on CuArrays (state fields + kernels).
- [ ] **Sanity checks**: u_K bounded, monotone approach to equilibrium for
      constant load, no spectral ringing at high k.

## 6. Phase 4 — Extended Burgers (Prony fit)

- [ ] **Prony fitting utility**: given (Δ, α, τ_L, τ_H) of the I&C/Faul–Jackson
      spectrum, fit N log-spaced branches (Δⱼ, τⱼ); return fit error. Lives in
      `src/material.jl` or a small utility file.
- [ ] **Choose default N** by fit-error vs. cost study (expect N ≈ 3–5).
- [ ] **Calibrated parameter presets** from I&C 2021 (their Table/section values)
      exposed as named constructors or documented defaults.

## 7. Phase 5 — Docs & caveats

- [ ] **Docstrings + docs page**: derivation of §2.3 (correspondence principle,
      compliance additivity), Maxwell as special case, parameter meaning
      (Δ, τ), and how this relates to I&C 2021.
- [ ] **Stated caveats**: (a) incompressible mode equation + v1-style
      compressibility scaling on η₁ only, not the EBM's frequency-dependent ν̃(s);
      (b) flat vs. spherical geometry; (c) Prony discretization error (report N);
      (d) laterally constant parameters only.
- [ ] **Update `AbstractMantle` docstring list** (`src/solidearth.jl:42-50`) to
      include `BurgersMantle`.

## 8. Deferred / open questions

- **Laterally variable Burgers parameters** — the v1 trick (effective viscosity
  field + scaled pseudodiff) has no proven analogue here yet; needs thought
  before attempting. Out of scope until §5 is done.
- **Interaction with FIAlgorithm built-in steppers** (FIBS3/FITsit5/FIEuler): the
  Burgers path is semi-implicit like the laterally-constant Maxwell path, so it
  bypasses the explicit RK steppers; revisit if the semi-implicit paths get
  unified with FIAlgorithm.
- **AD/Enzyme compatibility**: the AD roadmap (roadmap_ad_inversion.md) targets
  the explicit laterally-variable path first; the Burgers path adds state fields
  and a semi-implicit solve — if it ever becomes an AD target, the CN solve is
  closed-form so EnzymeRules should be straightforward. Not planned now.
- **Restart**: once JLD2 checkpointing lands (restart roadmap), u_K must be in
  the saved state — coordinate with that effort.
- **Ocean/rotational feedbacks with transient rheology**: no change expected
  (they couple through u only), but confirm once running.

---

## 9. References

- Ivins, E. R., Caron, L., Adhikari, S., Larour, E. (2021). *Notes on a
  compressible extended Burgers model of rheology.* Geophys. J. Int. 228(3),
  1975–1991. https://academic.oup.com/gji/article/228/3/1975/6414531
  — EBM creep function, Laplace-domain μ̃_EBM(s), Wolf-1985 half-space
  application, Heaviside/ramp subsidence benchmarks.
- Swierczek-Jereczek et al. (2024). *FastIsostasy v1.* GMD 17, 5263.
  https://gmd.copernicus.org/articles/17/5263/2024/
- Bueler, E., Lingle, C. S., Brown, J. (2007). Fourier collocation for the
  Lingle–Clark model. https://www.cambridge.org/core/product/identifier/S0260305500253974/type/journal_article
- Lingle, C. S., Clark, J. A. (1985). http://doi.wiley.com/10.1029/JC090iC01p01100
- Wolf, D. (1985). Half-space equations of motion used by I&C 2021.
- Faul, U., Jackson, I. — anelasticity relaxation spectrum underlying the EBM.
