function update_maskgrounded!(sim::Simulation)
    update_maskgrounded!(sim.now.maskgrounded, sim.now.H_af, sim.opts.transition)
    return nothing
end

function update_maskgrounded!(maskgrounded, H_af, ::SharpTransition)
    @. maskgrounded = H_af > 0
    return nothing
end

function update_maskgrounded!(maskgrounded, H_af, tr::SmoothTransition)
    @. maskgrounded = sheaviside(H_af, tr.eps)
    return nothing
end

get_maskgrounded(state, c) = get_maskgrounded(state, c, SharpTransition())
get_maskgrounded(state, c, tr::AbstractTransition) =
    get_maskgrounded(state.H_ice, state.z_b, state.z_ss, c, tr)

get_maskgrounded(H_ice, b, z_ss, c) =
    get_maskgrounded(H_ice, b, z_ss, c, SharpTransition())

function get_maskgrounded(H_ice, b, z_ss, c, tr::SharpTransition)
    return height_above_floatation(H_ice, b, z_ss, c, tr) .> 0
end

function get_maskgrounded(H_ice, b, z_ss, c, tr::SmoothTransition)
    return sheaviside.(height_above_floatation(H_ice, b, z_ss, c, tr), tr.eps)
end

get_maskocean(z_ss, b, maskgrounded) =
    get_maskocean(z_ss, b, maskgrounded, SharpTransition())

function get_maskocean(z_ss, b, maskgrounded, ::SharpTransition)
    return ((z_ss - b) .> 0) .& not.(maskgrounded)
end

function get_maskocean(z_ss, b, maskgrounded, tr::SmoothTransition)
    return sheaviside.(z_ss .- b, tr.eps) .* not.(maskgrounded)
end

function height_above_floatation(
    state::AbstractState,
    c::PhysicalConstants,
    tr::AbstractTransition = SharpTransition(),
)
    return height_above_floatation(state.H_ice, state.z_b, state.z_ss, c, tr)
end

function height_above_floatation(H_ice, z_b, z_ss, c)
    return height_above_floatation(H_ice, z_b, z_ss, c, SharpTransition())
end

function height_above_floatation(H_ice, z_b, z_ss, c, ::SharpTransition)
    return max.(H_ice .+ min.(z_b .- z_ss, 0) .* (c.rho_seawater / c.rho_ice), 0)
end

function height_above_floatation(H_ice, z_b, z_ss, c, tr::SmoothTransition)
    e = tr.eps
    return srelu.(H_ice .+ snegrelu.(z_b .- z_ss, e) .* (c.rho_seawater / c.rho_ice), e)
end

function update_maskocean!(sim)
    update_maskocean!(
        sim.now.maskocean,
        sim.now.z_ss,
        sim.now.z_b,
        sim.now.maskgrounded,
        sim.opts.transition,
    )
    return nothing
end

function update_maskocean!(maskocean, z_ss, z_b, maskgrounded, ::SharpTransition)
    @. maskocean = (z_ss - z_b) > 0
    @. maskocean = maskocean & not(maskgrounded)
    return nothing
end

function update_maskocean!(maskocean, z_ss, z_b, maskgrounded, tr::SmoothTransition)
    @. maskocean = sheaviside(z_ss - z_b, tr.eps) * not(maskgrounded)
    return nothing
end

function update_bedrock!(sim::Simulation, u)
    sim.now.u .= u
    @. sim.now.z_b = sim.ref.z_b + sim.now.ue + sim.now.u
    return nothing
end

function update_Haf!(sim::Simulation)
    update_Haf!(
        sim.now.H_af,
        sim.now.H_ice,
        sim.now.z_b,
        sim.now.z_ss,
        sim.c,
        sim.opts.transition,
    )
    return nothing
end

function update_Haf!(H_af, H_ice, z_b, z_ss, c, ::SharpTransition)
    @. H_af = max(H_ice + min(z_b - z_ss, 0) * c.rho_sw_ice, 0)
    return nothing
end

function update_Haf!(H_af, H_ice, z_b, z_ss, c, tr::SmoothTransition)
    e = tr.eps
    @. H_af = srelu(H_ice + snegrelu(z_b - z_ss, e) * c.rho_sw_ice, e)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the height above floatation `H_F` in the sense of
[adhikari_kinematic_2020](@citet) (their Eqs. 7 and 8),

    H_0 = (ρ_o/ρ_i) max(z_ss - z_b, 0)
    H_F = 𝒢 (H_ice - H_0)

with `𝒢 = sim.now.maskgrounded` the grounded-ice mask. This is *not* the same
field as `H_af`, which clamps the ice column at floatation with `max(⋅, 0)` and
is what feeds the ice load through `columnanom_ice!`: `H_F` is restricted
to the grounded domain but left signed inside it, because Adhikari's bookkeeping
needs cells that can *take up* ocean water (`H_F < 0`) to contribute to sea-level
fall.

Note that the two fields coincide as long as the ocean mask is evaluated
pointwise, as it is here: a cell is only ever grounded where `H_ice > H_0`. They
part ways once the ocean mask is repaired for connectivity (their Eq. 3), which
turns isolated below-floatation regions — subglacial troughs, proglacial lakes —
into land and so admits `H_F < 0`. `H_F` is therefore written as their Eq. (8)
rather than as an alias of `H_af`, so that the repair is the only change needed.
"""
function update_HF!(sim::Simulation)
    update_HF!(
        sim.now.H_F,
        sim.now.H_ice,
        sim.now.z_b,
        sim.now.z_ss,
        sim.now.maskgrounded,
        sim.c,
        sim.opts.transition,
    )
    return nothing
end

function update_HF!(H_F, H_ice, z_b, z_ss, maskgrounded, c, ::SharpTransition)
    @. H_F = maskgrounded * (H_ice - c.rho_sw_ice * max(z_ss - z_b, 0))
    return nothing
end

function update_HF!(H_F, H_ice, z_b, z_ss, maskgrounded, c, tr::SmoothTransition)
    e = tr.eps
    @. H_F = maskgrounded * (H_ice - c.rho_sw_ice * srelu(z_ss - z_b, e))
    return nothing
end
