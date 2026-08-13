# Transient creep: derivation

This page derives [`TransientCreepMantle`](@ref) step by step, from the steady
[`ViscousMantle`](@ref) response you may already be familiar with, up to the
closed-form solver FastIsostasy actually runs and the analytic solution used to
validate it. For the physical motivation (why transient creep matters, how it
compares to [ivins_notes_2021](@citet)'s results) see the [Transient creep](@ref)
worked example instead — this page is the theory behind it.

## 1. Where we start: the steady (Maxwell) response

FastIsostasy's semi-implicit solver works mode by mode in Fourier space. For a
wavenumber ``k = |\mathbf k|``, the steady-creep (`ViscousMantle`) equation is

```math
2\eta k \, \frac{d\hat u}{dt} = \hat F - \beta \hat u,
\qquad \beta = \rho g + D k^4,
```

where ``\hat u`` is the (Fourier-transformed) viscous displacement, ``\hat F``
the forcing (the load, transformed the same way), ``\eta`` the upper-mantle
viscosity and ``D k^4`` the lithosphere's flexural rigidity term. This comes
from the general viscoelastic half-space transfer function

```math
\hat u(s) = \frac{\hat F(s)}{\beta + 2k\,\tilde\mu(s)},
```

with ``\tilde\mu(s)`` the Laplace-domain shear modulus of whatever rheology sits
in the half-space [bueler_fast_2007](@citet). This is the **correspondence
principle**: solve the elastic problem, then replace the elastic modulus by its
Laplace-domain viscoelastic equivalent. A Newtonian dashpot has
``\tilde\mu(s) = \eta s``, and substituting that back gives exactly the ODE
above.

`ViscousMantle` is a *single* relaxation time per mode: the whole response
decays as one exponential, ``\hat u(t) = \hat u_{\mathrm{eq}}(1 - e^{-\beta
t/(2\eta k)})``. Real mantle rock does not do this — laboratory creep
experiments and geodetic observations (post-seismic flow, tidal response, the
first years after a rapid ice-mass change) all show a faster, partly-recoverable
*transient* (primary) creep on top of the steady one. Capturing that is what
`TransientCreepMantle` adds, without touching the elastic part of the model
(the Farrell convolution) at all — see the note at the end of §2.

## 2. The Burgers body: one Kelvin branch

The classic rheological model for transient creep is the **Burgers body**: a
Maxwell dashpot (viscosity ``\eta_1``) in **series** with a Kelvin–Voigt element
(a spring ``\mu_2`` and dashpot ``\eta_2`` in *parallel*). Elements in series
carry the same stress and their compliances (strain per unit stress) add, so the
Laplace-domain compliance of the whole body is

```math
\frac{1}{\tilde\mu(s)} = \underbrace{\frac{1}{\eta_1 s}}_{\text{Maxwell dashpot}}
  + \underbrace{\frac{1}{\mu_2 + \eta_2 s}}_{\text{Kelvin-Voigt element}}.
```

Two numbers describe the Kelvin branch relative to the Maxwell one:

```math
\Delta = \frac{\mu_1}{\mu_2} \quad \text{(relaxation strength)}, \qquad
\tau = \frac{\eta_2}{\mu_2} \quad \text{(Kelvin/retardation time, in years)},
```

where ``\mu_1`` is the *unrelaxed* shear modulus of the whole body — a
rheological parameter fixing the scale of ``\mu_2 = \mu_1/\Delta``, **not** the
same ``\mu`` used by the Farrell elastic Green's function. The two stay
conceptually and numerically separate: `TransientCreepMantle` only ever adds a
*transient anelastic band* on top of whatever the elastic convolution already
computes. This is also why ``\Delta \to 0`` (equivalently ``\mu_2 \to \infty``)
must reduce the model to plain `ViscousMantle` exactly — the Kelvin branch locks
solid and stops contributing.

## 3. From one branch to a Prony series

Ivins & Caron's extended-Burgers model (EBM) replaces the single Kelvin branch
with a *continuous* spectrum of relaxation times [ivins_notes_2021](@citet).
FastIsostasy approximates that spectrum with ``N`` discrete Kelvin branches (a
Prony series) in series with the same Maxwell dashpot. All branches see the same
stress, so their displacements simply add:

```math
\hat u = \hat u_M + \sum_{j=1}^N \hat u_K^{(j)}.
```

Writing out each branch's equation of motion (§1's correspondence principle
applied per branch) gives the coupled ODE system FastIsostasy actually solves:

```math
2\eta_1 k\, \frac{d\hat u_M}{dt}
  = \hat F - \beta \Big(\hat u_M + \sum_{i=1}^N \hat u_K^{(i)}\Big),
```

```math
2\eta_{2j} k\, \frac{d\hat u_K^{(j)}}{dt}
  = \hat F - \beta \Big(\hat u_M + \sum_{i=1}^N \hat u_K^{(i)}\Big) - 2\mu_{2j} k\, \hat u_K^{(j)},
  \qquad j = 1, \dots, N.
```

Every coefficient depends only on ``k``, so — exactly as for `ViscousMantle` —
this is a plain per-mode Fourier collocation, with no closure problem to solve
for laterally-variable parameters (that generalisation is a separate,
unresolved question; see §6). Setting every ``\hat u_K^{(j)} \equiv 0`` recovers
the `ViscousMantle` equation term for term, which is why the ``\Delta_j \to 0``
limit is such a strong correctness check.

### Fitting a continuous spectrum: `fit_prony_series`

Ivins & Caron discretise a *continuous* absorption-band density,

```math
F(\tau) = \frac{\alpha\, \tau^{\alpha-1}}{\tau_H^\alpha - \tau_L^\alpha},
\qquad \tau_L \le \tau \le \tau_H,
```

a probability density (``\int_{\tau_L}^{\tau_H} F\,d\tau = 1``), so that a
total relaxation strength ``\Delta`` spread over the band contributes
``\Delta \int_{\tau_L}^{\tau_H} F(\tau)\big(1-e^{-t/\tau}\big)\,d\tau`` to the
creep function. [`fit_prony_series`](@ref) turns this into the ``N``-branch sum
above *without* a nonlinear fit: because ``F`` is a power law, its CDF and
first moment are elementary, so partitioning ``[\tau_L, \tau_H]`` into ``N``
log-spaced bins gives closed-form ``\Delta_j`` (the exact probability mass of
bin ``j``, so ``\sum_j \Delta_j = \Delta`` for any ``N``) and ``\tau_j`` (the
``F``-weighted mean retardation time within it). [`BurgersMantle`](@ref) and
[`ExtendedBurgersMantle`](@ref) wrap this — and the plain ``N=1`` case — into
ready-to-use [`TransientCreepMantle`](@ref) constructors.

`fit_prony_series` also returns `fit_error`: the largest relative deviation
between the ``N``-branch sum and the continuous integral above, evaluated by
Gauss-Legendre quadrature over a log-spaced grid of ``t \in [\tau_L,\tau_H]``.
For ``\alpha=1/2`` and a two-decade band (``\tau_H/\tau_L = 100``), measured
`fit_error` falls from 17% at ``N=1`` to under 1% by ``N=5``:

| ``N``       | 1     | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|:-----------:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| `fit_error` | 16.6% | 5.6% | 2.5% | 1.4% | 0.9% | 0.6% | 0.5% | 0.4% |

consistent with the ``N \approx 3\text{-}5`` rule of thumb in
[ivins_notes_2021](@citet). There is no universal default ``N``: pick it from
`fit_error` at your own ``(\alpha, \tau_L, \tau_H)``.

## 4. Time stepping: why Crank–Nicolson

Kelvin retardation times can be years to decades while Maxwell relaxation times
are centuries to millennia — the system above is **stiff**. An explicit scheme
would need timesteps set by the *fastest* branch even though the interesting
dynamics happen on the slower ones, so FastIsostasy uses semi-implicit
Crank–Nicolson (CN) instead, exactly like the laterally-constant `ViscousMantle`
path it generalises. CN is unconditionally stable for this problem (A-stable):
no combination of step size and branch parameters makes it blow up. It is not
*L-stable*, though, meaning very stiff modes decay with a sign-alternating
amplification factor close to ``-1`` rather than being damped to ``0`` in one
step — a well-known, generally accepted CN trade-off, checked explicitly for
this solver in the test suite (§7).

Label the ``N+1`` unknowns per mode with an index ``i``: ``i = 0`` for the
Maxwell branch and ``i = 1, \dots, N`` for the Kelvin branches, with

```math
(d_0, e_0) = (2\eta_1 k,\ 0), \qquad
(d_j, e_j) = (2\eta_{2j} k,\ 2\mu_{2j} k) = (\tau_j C_j,\ C_j), \quad C_j = 2\mu_{2j}k.
```

Every branch's equation then has the same shape,
``d_i\, d\hat u_i/dt = \hat F - \beta \hat S - e_i \hat u_i``, with
``\hat S = \sum_k \hat u_k`` the total. Crank–Nicolson (freezing ``\hat F`` over
the half-step, as the load is piecewise-linear between save points) turns this
into, for each ``i``,

```math
g_i\, \hat u_i^{n+1} + a\beta\, \hat S^{n+1} = r_i,
\qquad g_i = d_i + a e_i, \qquad a = \frac{\Delta t}{2},
```

```math
r_i = (d_i - a e_i)\, \hat u_i^n - a\beta\, \hat S^n + \Delta t\, \hat F.
```

## 5. An O(N) closed form

The system in §4 looks like it needs an ``(N+1)\times(N+1)`` solve per Fourier
mode — but every branch couples to every other one **only** through their
shared sum ``\hat S``, never pairwise. In matrix form this is a *diagonal
matrix plus a rank-1 correction*, which Sherman–Morrison inverts in closed form.
Summing the ``g_i \hat u_i^{n+1} + a\beta \hat S^{n+1} = r_i`` line over all
``i`` and solving for ``\hat S^{n+1}`` gives

```math
\hat S^{n+1} = \frac{\displaystyle\sum_i r_i/g_i}
                     {1 + a\beta \displaystyle\sum_i 1/g_i},
\qquad
\hat u_i^{n+1} = \frac{r_i - a\beta\, \hat S^{n+1}}{g_i}.
```

This is ``O(N)``, not a per-mode matrix factorisation: two passes over the
branches (one to accumulate the sums, one to finalise each ``\hat u_i^{n+1}``),
each pass just elementwise arithmetic over the spectral grid. Two
simplifications make the implementation cheaper still:

- For ``i = 0`` (Maxwell), ``(d_0 - a e_0)/g_0 = 1`` exactly, and
  ``\hat u_M^n = \hat S^n - \sum_j \hat u_K^{(j)n}`` by linearity — so the
  Maxwell branch's own state never has to be materialised. Only ``\hat S^n``
  (the Fourier transform of the state array `u`, which already *is*
  ``u_M + \sum_j u_K^{(j)}``) and each Kelvin branch's own state are needed.
- For branch ``j``, ``(d_j - a e_j)/g_j = (\tau_j - a)/(\tau_j + a)`` — the
  branch's own shear modulus ``\mu_{2j}`` cancels between numerator and
  denominator, leaving a *scalar* (not a spectral field) coefficient. Only
  ``1/g_j`` in the ``\hat S^{n+1}`` sum needs the ``k``-dependent field
  ``C_j``.

Setting ``N = 1`` in the formulas above reproduces `ViscousMantle`'s
Crank–Nicolson update once ``\Delta \to 0``, and reproduces a hand-derived
closed-form ``2\times 2`` solve exactly when ``N = 1`` and ``\Delta`` is
anything else — both checked in the test suite, alongside a direct comparison
against `ViscousMantle` for two *identical* Kelvin branches, which must equal
one branch of doubled relaxation strength (compliances add).

## 6. Scope: laterally constant parameters only

Every coefficient in §§4–5 depends only on ``k``, which is what makes the
Sherman–Morrison solve exact — it relies on ``\mu_1``, ``\Delta_j`` and
``\tau_j`` being the same at every grid point. `ViscousMantle`'s laterally
*variable* case (spatially varying viscosity and lithosphere thickness) uses a
different trick — an effective-viscosity field folded into a scaled pseudo-
differential operator [swierczek-jereczek_fastisostasy_2024](@citet) — and
that trick has no proven analogue for the coupled ``(N+1)``-field system yet.
`TransientCreepMantle` therefore requires
[`LaterallyConstantLithosphere`](@ref) or [`RigidLithosphere`](@ref), and
errors clearly otherwise.

## 7. An exact solution for a suddenly imposed disc load

To validate the ``N = 1`` solver independently of the discretisation itself, it
helps to have a closed-form solution to compare against. For a disc load turned
on as a step at ``t = 0`` and held constant (a Heaviside forcing,
``\hat F(t) = \hat F_0`` for ``t \geq 0``), each Fourier mode's step response can
be found by inverting the transfer function from §1 directly, without
discretising time at all.

With ``\hat F(s) = \hat F_0/s`` (the Laplace transform of a Heaviside) and the
Burgers compliance from §2, the transfer function ``\hat u(s) = \hat
F(s)/(\beta + 2k\tilde\mu(s))`` has three poles: ``s = 0`` (from the ``1/s`` of
the step) and the two roots of a quadratic,

```math
2k\eta_1\eta_2\, s^2 + \big[\beta(\eta_1+\eta_2) + 2k\eta_1\mu_2\big]\, s + \beta\mu_2 = 0,
```

both real and negative for physically sensible parameters (the poles of a
passive, dissipative system). Call them ``s_a`` and ``s_b``, and write
``a_{\mathrm{coef}} = 2k\eta_1\eta_2`` for the quadratic's leading coefficient.
Partial-fractioning ``\hat u(s)`` and inverting term by term gives, for
``t \geq 0``,

```math
\hat u(t) = \hat u_{\mathrm{eq}}\Big[\, 1 - A_a e^{s_a t} - A_b e^{s_b t} \,\Big],
\qquad \hat u_{\mathrm{eq}} = \frac{\hat F_0}{\beta},
```

```math
A_i = -\frac{\beta Q_i}{a_{\mathrm{coef}}}, \qquad
Q_i = \frac{\mu_2 + (\eta_1+\eta_2)s_i}{s_i(s_i - s_{i'})} \quad (i' \ne i).
```

At ``t=0`` this gives ``\hat u = 0`` (since ``A_a + A_b = 1``, from
``\hat u(0)=0``); as ``t \to \infty``, both exponentials vanish and
``\hat u \to \hat u_{\mathrm{eq}}`` — the *same* equilibrium as plain
`ViscousMantle`, consistent with §2's ``\Delta \to 0`` argument (a Kelvin branch
stores no permanent deformation). Superposing this per-mode result over the
Hankel transform of a disc of radius ``R_0`` (the same construction
`ViscousMantle`'s analytic solution already uses, see `analytic_solution` in
`src/analytic_solutions.jl`) gives the full spatial disc-load solution used to
validate the solver.

!!! note "Verified independently before landing"
    The formula above was checked against a direct 4th-order Runge–Kutta
    integration of the two-state ODE system (§2) at several wavenumbers and
    times, to machine-level agreement, before being written into
    `src/analytic_solutions.jl` — the algebra from transfer function to
    time-domain closed form has several sign-convention pitfalls (which
    quantity is 0 at ``t=0`` versus at ``t \to \infty``) that a purely symbolic
    derivation can get backwards without anything as obviously wrong-looking as
    a divergence.

## 8. Validation summary

- **Steady-creep limit.** ``\Delta_j \to 0`` reproduces `ViscousMantle` to near
  machine precision, for both ``N=1`` and ``N>1`` — a permanent regression test.
- **Analytic disc load.** The ``N=1`` solver matches the closed form of §7 along
  a transect, at several times, to the same tolerance historically accepted for
  `ViscousMantle` against this same analytic solution (the residual is
  dominated by the finite periodic FFT domain versus the analytic infinite
  half-space, not by solver error) — and the error shrinks as the timestep
  shrinks, confirming the discretisation converges.
- **GPU.** The general-``N`` solve runs on `CuArray` (`N=1` and `N>1`), matching
  the CPU result within tolerance.
- **Sanity checks.** The Kelvin state stays bounded and finite; a load held
  constant after a short ramp is approached monotonically (no overshoot,
  consistent with §4's real, negative eigenvalues); and a spatially rough
  (high-``k``) load, stepped with a timestep deliberately large relative to the
  Kelvin time, relaxes smoothly rather than ringing.
- **Qualitative reproduction of [ivins_notes_2021](@citet).** See the
  [Transient creep](@ref) worked example.
- **Extended Burgers fit.** [`fit_prony_series`](@ref)'s `N`-branch
  approximation converges as `N` grows (measured `fit_error`, §3); `N ≈ 3-5`
  lands within a few percent for a representative two-decade spectrum,
  matching [ivins_notes_2021](@citet)'s rule of thumb.

## 9. Caveats

- **Compressibility.** The same approximation as the laterally-variable
  `ViscousMantle` path: a scalar compressibility correction is applied to
  ``\eta_1`` only. The EBM's frequency-dependent effective Poisson ratio
  ``\tilde\nu(s)`` is not implemented.
- **Geometry.** A flat half-space, not a spherical, self-gravitating Earth —
  the same approximation `ViscousMantle` makes, and the one
  [ivins_notes_2021](@citet) also use in their §4 loading study.
- **Discretisation of the relaxation spectrum.** ``N`` discrete branches
  approximate what is physically a continuous spectrum
  [ivins_notes_2021](@citet). ``N=1`` (the classic Burgers body) captures the
  *amplitude* and *decay* of the transient enhancement but not the detailed
  shape of a continuous spectrum. [`fit_prony_series`](@ref) (§3) fits
  ``N \approx 3\text{-}5`` branches to any given ``(\Delta, \alpha, \tau_L,
  \tau_H)`` band; what remains open is a *literature-calibrated* preset — the
  actual published values from [ivins_notes_2021](@citet) have not yet been
  entered into the package (fastisostasy-roadmap/burgers.md §6).
- **Laterally constant parameters only** (§6).
- **Requires** [`EulerIntegrator`](@ref) (a fixed step; the semi-implicit solve
  has no adaptive-step variant) with `SolverOptions(fft = ComplexFFTBackend())`
  — `RealFFTBackend` is not yet implemented for this rheology.

## 10. References

- [bueler_fast_2007](@citet)
- [lingle_numerical_1985](@citet)
- [ivins_notes_2021](@citet)
- [swierczek-jereczek_fastisostasy_2024](@citet)
