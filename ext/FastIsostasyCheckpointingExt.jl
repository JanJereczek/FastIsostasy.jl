module FastIsostasyCheckpointingExt

# Checkpointed reverse-mode adjoint of the forward model.
#
# SKELETON (roadmap Phase 5). Triggered by `Checkpointing` + `Enzyme`. Will
# provide the periodic-schedule checkpointing that backs
# `gradient!(g, prob, θ)` for `AdjointMode`: forward recording of the accepted
# (t, dt) sequence and per-interval state snapshots, then per-step
# `Enzyme.autodiff(Reverse, advance_step!, …)` with the dt-sequence frozen.

using Checkpointing: Checkpointing
using Enzyme: Enzyme
import FastIsostasy

end # module
