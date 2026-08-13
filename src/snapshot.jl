# =============================================================================
# State snapshots: capture / restore the full mutated state of a `Simulation`.
#
# A forward run mutates three places: `sim.now` (a `CurrentState` — all the
# diagnostic/integrated arrays plus scalars `count_sparse_updates`, `z_bsl`,
# `V_af/V_pov/V_den`, `delta_V`), the barystatic sea level `sim.sealevel.bsl`
# (`z`, `A`, `residual`, recursively for `CombinedBSL`), and the clock
# `sim.timer.t`. Everything else (domain, constants, parameters, plans, ice
# snapshots) is fixed across a run.
#
# `snapshot!(buf, sim)` copies all of it into a preallocated buffer; `restore!`
# writes it back. Together they let the Phase-5 adjoint recompute a forward
# interval from a saved checkpoint, and are shared with the JLD2 restart
# machinery. Buffers are reused via in-place copies, so checkpointing allocates
# only once (at `StateSnapshot(sim)`). Timer *logging* vectors (`t_vec`,
# `t_computation`) are intentionally not captured — they are instrumentation,
# don't affect the physical trajectory, and `t_computation!` is AD-inactive.
# =============================================================================

"""
    StateSnapshot(sim)

Allocate a buffer holding a full copy of `sim`'s mutable state (its `CurrentState`,
its barystatic-sea-level object, and the clock). Reuse it across many
`snapshot!`/`restore!` calls.
"""
mutable struct StateSnapshot{CS,BSL,T}
    now::CS
    bsl::BSL
    timer_t::T
end

StateSnapshot(sim) =
    StateSnapshot(deepcopy(sim.now), deepcopy(sim.sealevel.bsl), sim.timer.t)

# Copy every mutated field of a `CurrentState` (arrays in place, the nested
# array-only structs `ColumnAnomalies`/`KinematicBSL` array-by-array, scalars by
# assignment). The nested structs must go through `copyto!` like everything else:
# `setfield!`-ing them would make `dst` alias `src`'s arrays, and a restore would
# then silently be a no-op.
function _copy_state!(dst::CurrentState, src::CurrentState)
    for f in fieldnames(CurrentState)
        sv = getfield(src, f)
        if sv isa AbstractArray
            copyto!(getfield(dst, f), sv)
        elseif sv isa ColumnAnomalies || sv isa KinematicBSL
            dnested = getfield(dst, f)
            for cf in fieldnames(typeof(sv))
                copyto!(getfield(dnested, cf), getfield(sv, cf))
            end
        else
            setfield!(dst, f, sv)          # scalar (`T` or `Int`)
        end
    end
    return nothing
end

# Copy the mutated scalars of a barystatic-sea-level object (`z`, `A`,
# `residual` — all `<:Real`), recursing through the sub-BSLs of a `CombinedBSL`.
# Reference data (`ref`, interpolators, `z_vec`, `mcp_opts`) is fixed across a run.
function _copy_bsl!(dst, src)
    for f in fieldnames(typeof(src))
        sv = getfield(src, f)
        if sv isa Real
            setfield!(dst, f, sv)
        elseif sv isa AbstractBSL
            _copy_bsl!(getfield(dst, f), sv)
        end
    end
    return nothing
end

"""
    snapshot!(buf::StateSnapshot, sim) -> buf

Copy `sim`'s current mutable state into the preallocated `buf`.
"""
function snapshot!(buf::StateSnapshot, sim)
    _copy_state!(buf.now, sim.now)
    _copy_bsl!(buf.bsl, sim.sealevel.bsl)
    buf.timer_t = sim.timer.t
    return buf
end

"""
    restore!(sim, buf::StateSnapshot) -> sim

Write the state stored in `buf` back into `sim`, so a forward run resumed from
here reproduces the trajectory captured at `snapshot!` time.
"""
function restore!(sim, buf::StateSnapshot)
    _copy_state!(sim.now, buf.now)
    _copy_bsl!(sim.sealevel.bsl, buf.bsl)
    sim.timer.t = buf.timer_t
    return sim
end
