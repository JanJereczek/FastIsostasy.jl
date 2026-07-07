module FastIsostasyEnzymeExt

# Enzyme-based differentiation of `loss(prob, θ)`.
#
# SKELETON (roadmap Phase 2). This module will provide:
#   - `EnzymeRules.inactive` for `t_computation!` / printing / NetCDF writers,
#   - forward/reverse `EnzymeRules` for planned-FFT `mul!` application,
#   - `gradient!(g, prob, θ)` dispatching on `prob.diffmode`
#     (`TangentMode` → `Enzyme.autodiff(Forward, …)`,
#      `AdjointMode` → checkpointed reverse, see FastIsostasyCheckpointingExt).

using Enzyme: Enzyme
import FastIsostasy: gradient!, AbstractInversion

function gradient!(g, prob::AbstractInversion, θ)
    error("Enzyme-based `gradient!` is not implemented yet (roadmap Phase 2). " *
          "This extension is a skeleton.")
end

end # module
