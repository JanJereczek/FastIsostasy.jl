# =============================================================================
# Progress reporting for forward runs (`run!`).
#
# Driven from the stepper's accept branch, so the display advances with the
# integration itself rather than only at output times — a run that writes no
# output still gets a live bar.
# =============================================================================

# Resolution of the bar: the simulated time span is mapped onto this many ticks,
# so `dt`-varying adaptive steps still produce a monotone counter.
const PROGRESS_TICKS = 1000

"""
$(TYPEDSIGNATURES)

Wall-clock-throttled progress display for a forward [`run!`](@ref).

Refreshes at most once every `dt_walltime` seconds. The throttle is ours rather
than `ProgressMeter`'s own: that package builds its `showvalues` *eagerly* at
the call site, and each diagnostic below is an `extrema` reduction over a
grid-sized field — a device synchronisation on GPU. Handing them straight to
the bar would therefore recompute all of them on every accepted step, whether
or not the bar redrew. `ForwardProgress` touches the state only when it is
actually about to print, so the cost is bounded by wall time instead of by step
count.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ForwardProgress
    "the underlying `ProgressMeter.Progress` bar"
    bar::Progress
    "minimum wall time between two refreshes, in seconds"
    dt_walltime::Float64
    "`time()` at the last refresh"
    t_wall_last::Float64
    "start of the simulated time span"
    t_start::Float64
    "length of the simulated time span (guaranteed `> 0`)"
    t_span::Float64
end

function ForwardProgress(sim; dt_walltime = sim.opts.dt_walltime)
    t_start, t_end = Float64(sim.timer.t_span[1]), Float64(sim.timer.t_span[2])
    # `dt = 0` on the bar itself: `report_progress!` has already decided that it
    # is time to redraw, so a second throttle could only swallow that refresh.
    bar = Progress(
        PROGRESS_TICKS;
        dt = 0.0,
        desc = "Forward run: ",
        color = :green,
        showspeed = false,
    )
    # Backdate the throttle so the very first accepted step draws the bar: a run
    # shorter than `dt_walltime` would otherwise never print at all, and
    # `ProgressMeter` suppresses its completion line unless the bar has been
    # drawn at least once.
    return ForwardProgress(
        bar,
        Float64(dt_walltime),
        time() - Float64(dt_walltime),
        t_start,
        max(t_end - t_start, eps(Float64)),
    )
end

# Bar position for a simulated time. Capped one tick below the end so that
# `finish_progress!` is the only thing that ever completes the bar — reaching
# `PROGRESS_TICKS` twice would print the completion line twice.
progress_tick(p::ForwardProgress, t) = clamp(
    round(Int, PROGRESS_TICKS * (Float64(t) - p.t_start) / p.t_span),
    0,
    PROGRESS_TICKS - 1,
)

_sig(x) = round(Float64(x); sigdigits = 4)

# `extrema` is one pass over the field and, on GPU, one synchronisation — hence
# the wall-clock throttle in `report_progress!` that gates every call to this.
function _extrema_str(field)
    lo, hi = extrema(field)
    return "$(_sig(lo)) … $(_sig(hi))"
end

function progress_values(sim, integ)
    return [
        ("sim time [yr]", _sig(integ.t)),
        ("time step [yr]", _sig(integ.dt)),
        ("viscous displacement u [m]", _extrema_str(sim.now.u)),
        ("elastic displacement ue [m]", _extrema_str(sim.now.ue)),
        ("sea-surface perturbation dz_ss [m]", _extrema_str(sim.now.dz_ss)),
        ("barystatic sea level bsl [m]", _sig(sim.now.z_bsl)),
    ]
end

"""
$(TYPEDSIGNATURES)

Refresh the forward-run progress display from the integrator's current state,
if at least `dt_walltime` seconds have passed since the last refresh. A
`nothing` progress (a silent run, or a manually driven `step!` loop) is a no-op
that costs nothing — the state is never touched.
"""
function report_progress!(p::ForwardProgress, integ)
    t_wall = time()
    t_wall - p.t_wall_last < p.dt_walltime && return nothing
    p.t_wall_last = t_wall
    # `ignore_predictor` disables ProgressMeter's own "has enough time passed?"
    # heuristic: the wall-clock check above is already the authority, and the
    # bar's `dt` is 0 precisely so it never second-guesses it.
    update!(
        p.bar,
        progress_tick(p, integ.t);
        showvalues = progress_values(integ.p, integ),
        ignore_predictor = true,
    )
    return nothing
end

report_progress!(::Nothing, integ) = nothing

# Force a last refresh so the bar ends on the true final state rather than on
# whatever the throttle last let through, then close it.
function finish_progress!(p::ForwardProgress, integ)
    update!(
        p.bar,
        PROGRESS_TICKS;
        showvalues = progress_values(integ.p, integ),
        ignore_predictor = true,
    )
    finish!(p.bar)     # no-op once the update above completed the bar
    return nothing
end

finish_progress!(::Nothing, integ) = nothing

# Whether `run!`'s per-output messages should be printed. They are suppressed
# while a bar is running: the bar redraws in place, so interleaved `println`s
# would leave a trail of half-finished bars, and the simulation time they report
# is already on the bar.
verbose_log(sim, ::Nothing) = sim.opts.verbose
verbose_log(sim, ::ForwardProgress) = false
