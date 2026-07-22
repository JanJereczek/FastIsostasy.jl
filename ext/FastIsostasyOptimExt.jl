module FastIsostasyOptimExt

# Optim.jl driver for inversions. `solve!` wraps `loss` (objective) and
# `gradient!` (from the Enzyme extension) into an `Optim.optimize` call. The
# objective/gradient plumbing is complete here; it becomes runnable once the
# Enzyme extension provides `gradient!` (roadmap Phase 2).

using Optim: Optim, optimize, Options, LBFGS, only_fg!
import FastIsostasy: solve!, loss, loss_and_gradient!, AbstractInversion

"""
    solve!(prob, θ0; optimizer = LBFGS(), iterations = 100, kwargs...)

Minimise `loss(prob, ·)` from the initial guess `θ0` using Optim. Returns the
`Optim.OptimizationResults`. Objective and gradient are supplied jointly by
`loss_and_gradient!(g, prob, θ)` (Enzyme extension, so that extension must also be
loaded) via `Optim.only_fg!`: under `TangentMode` the primal is a byproduct of the
same forward passes that compute the gradient, so this avoids the extra `loss`-only
evaluation a separate `f`/`g!` pair would cost every iteration.
"""
function solve!(prob::AbstractInversion, θ0;
        optimizer = LBFGS(), iterations::Int = 100, kwargs...)
    function fg!(F, G, θ)
        if G !== nothing
            l = loss_and_gradient!(G, prob, θ)
            return F === nothing ? nothing : l
        elseif F !== nothing
            return loss(prob, θ)
        end
        return nothing
    end
    return optimize(only_fg!(fg!), copy(θ0), optimizer,
        Options(; iterations = iterations, kwargs...))
end

end # module
