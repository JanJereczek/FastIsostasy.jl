function update_maskgrounded!(sim::Simulation)
    sim.now.maskgrounded .= sim.now.H_af .> 0
    return nothing
end

get_maskgrounded(state, c) = height_above_floatation(state, c) .> 0

function get_maskgrounded(H_ice, b, z_ss, c)
    return height_above_floatation(H_ice, b, z_ss, c) .> 0
end

function get_maskocean(z_ss, b, maskgrounded)
    return ((z_ss - b) .> 0) .& not.(maskgrounded)
end

function height_above_floatation(state::AbstractState, c::PhysicalConstants)
    return height_above_floatation(state.H_ice, state.z_b,
        state.z_ss, c)
end

function height_above_floatation(H_ice, b, z_ss, c)
    return max.(H_ice .+ min.(b .- z_ss, 0), 0) .* (c.rho_seawater / c.rho_ice)
end

function update_maskocean!(sim)
    @. sim.now.maskocean = (sim.now.z_ss - sim.now.z_b) > 0
    @. sim.now.maskocean = sim.now.maskocean .& not.(sim.now.maskgrounded)
end

function update_bedrock!(sim::Simulation, u)
    sim.now.u .= u
    @. sim.now.z_b = sim.ref.z_b + sim.now.ue + sim.now.u
    return nothing
end

function update_Haf!(sim::Simulation)
    @. sim.now.H_af = max(sim.now.H_ice + min(sim.now.z_b - sim.now.z_ss, 0), 0) 
    sim.now.H_af .*= sim.c.rho_sw_ice
    return nothing
end