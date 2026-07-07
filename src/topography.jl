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

get_maskgrounded(H_ice, b, z_ss, c) = get_maskgrounded(H_ice, b, z_ss, c, SharpTransition())

function get_maskgrounded(H_ice, b, z_ss, c, tr::SharpTransition)
    return height_above_floatation(H_ice, b, z_ss, c, tr) .> 0
end

function get_maskgrounded(H_ice, b, z_ss, c, tr::SmoothTransition)
    return sheaviside.(height_above_floatation(H_ice, b, z_ss, c, tr), tr.eps)
end

get_maskocean(z_ss, b, maskgrounded) = get_maskocean(z_ss, b, maskgrounded, SharpTransition())

function get_maskocean(z_ss, b, maskgrounded, ::SharpTransition)
    return ((z_ss - b) .> 0) .& not.(maskgrounded)
end

function get_maskocean(z_ss, b, maskgrounded, tr::SmoothTransition)
    return sheaviside.(z_ss .- b, tr.eps) .* not.(maskgrounded)
end

function height_above_floatation(state::AbstractState, c::PhysicalConstants,
    tr::AbstractTransition = SharpTransition())
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
    update_maskocean!(sim.now.maskocean, sim.now.z_ss, sim.now.z_b, sim.now.maskgrounded,
        sim.opts.transition)
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
    update_Haf!(sim.now.H_af, sim.now.H_ice, sim.now.z_b, sim.now.z_ss, sim.c,
        sim.opts.transition)
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
