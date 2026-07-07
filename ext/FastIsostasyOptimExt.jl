module FastIsostasyOptimExt

# Optim.jl driver for inversions. `solve!` wraps `loss` (objective) and
# `gradient!` (from the Enzyme extension) into an `Optim.optimize` call. The
# objective/gradient plumbing is complete here; it becomes runnable once the
# Enzyme extension provides `gradient!` (roadmap Phase 2).

using Optim: Optim, optimize, Options, LBFGS
import FastIsostasy: solve!, loss, gradient!, AbstractInversion

"""
    solve!(prob, θ0; optimizer = LBFGS(), iterations = 100, kwargs...)

Minimise `loss(prob, ·)` from the initial guess `θ0` using Optim. Returns the
`Optim.OptimizationResults`. The gradient is supplied by `gradient!(g, prob, θ)`
(Enzyme extension), so that extension must also be loaded.
"""
function solve!(prob::AbstractInversion, θ0;
        optimizer = LBFGS(), iterations::Int = 100, kwargs...)
    f(θ) = loss(prob, θ)
    g!(g, θ) = (gradient!(g, prob, θ); g)
    return optimize(f, g!, copy(θ0), optimizer,
        Options(; iterations = iterations, kwargs...))
end

end # module
