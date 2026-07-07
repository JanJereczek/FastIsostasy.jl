# Time integration

FastIsostasy integrates the viscous displacement forward in time with a small,
self-contained set of explicit Runge–Kutta methods. They live in
[`src/integrators.jl`](https://github.com/JanJereczek/FastIsostasy.jl/blob/main/src/integrators.jl)
and carry **no external ODE dependency** — earlier versions relied on
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), which has now
been dropped.

## Available algorithms

The algorithm is selected through [`DiffEqOptions`](@ref) inside
[`SolverOptions`](@ref). All three types share a single, generic in-place
stepper that works on CPU and GPU arrays alike:

| Algorithm   | Order | Adaptive | Notes                                                     |
|:------------|:-----:|:--------:|:----------------------------------------------------------|
| `FIBS3()`   |   3   |   yes    | Bogacki–Shampine 3(2), FSAL. **Default.**                 |
| `FITsit5()` |   5   |   yes    | Tsitouras 5(4), FSAL. Higher order for smooth problems.   |
| `FIEuler()` |   1   |    no    | Fixed step (requires `dt_min`); for implicit-style setups.|

```julia
opts = SolverOptions(diffeq = DiffEqOptions(alg = FIBS3(), reltol = 1f-5))
```

The adaptive methods use a scaled RMS error norm and a Hairer PI step-size
controller, matching the behaviour of the previous OrdinaryDiffEq solvers, so a
given `reltol` yields a comparable number of steps.

## Why the switch, and what it costs

The right-hand side of the FastIsostasy ODE (FFTs and convolutions) dominates the
cost of a step, so the fair comparison metric is the **number of RHS
evaluations** — if two backends need the same number of RHS calls, their wall
times match. The built-in stepper avoids the dense-output and interpolation work
that a general-purpose library performs, so it actually needs *fewer* RHS
evaluations while producing the same displacement field.

### Head-to-head benchmark

Analytic cylinder-load problem (128 × 128 grid, 50 kyr, `reltol = 1e-5`,
`Float32`), comparing the former OrdinaryDiffEq solvers against the built-in
`FIAlgorithm` equivalents. Reproduce with
[`benchmarks/benchmark_integrators.jl`](https://github.com/JanJereczek/FastIsostasy.jl/blob/main/benchmarks/benchmark_integrators.jl).

| Method | Backend       | Time [s] | Alloc [MiB] | RHS evals | Speed-up | Δu vs previous |
|:-------|:--------------|---------:|------------:|----------:|---------:|---------------:|
| BS3    | OrdinaryDiffEq |   0.890 |        1.85 |      1013 |    –     | –              |
| BS3    | **FIBS3**      |   0.686 |        0.90 |       648 | **1.30×**| 3.3 × 10⁻⁴ m   |
| Tsit5  | OrdinaryDiffEq |   3.071 |        2.11 |      5645 |    –     | –              |
| Tsit5  | **FITsit5**    |   1.779 |        1.11 |      2364 | **1.73×**| 2.0 × 10⁻³ m   |
| Euler  | OrdinaryDiffEq |   0.523 |        1.52 |       505 |    –     | –              |
| Euler  | **FIEuler**    |   0.527 |        0.70 |       500 |   0.99×  | 6.3 × 10⁻⁵ m   |

Takeaways:

- The built-in adaptive steppers are **1.3× (BS3) to 1.7× (Tsit5) faster** and
  allocate about **half** as much, mainly because they skip the library's
  dense-output bookkeeping.
- `FIEuler` matches the previous fixed-step Euler to machine-level accuracy at
  the same speed.
- The final displacement field differs by at most a few millimetres on a
  ~279 m field — well within the `reltol = 1e-5` error floor (≈ 3 × 10⁻³ m), so
  the solutions are physically identical.

For this problem `FIBS3` is the sweet spot: its cheaper embedded pair takes far
fewer steps than `FITsit5`, whose higher order is wasted because the step size
is limited by the non-smooth ice-load forcing rather than by local accuracy.

!!! note "Reproducibility"
    Absolute timings depend on the machine; the RHS-evaluation counts and field
    differences are the hardware-independent figures. The numbers above were
    measured on the development machine at commit time.
