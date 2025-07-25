# Functions to compute/update density-scaled column anomalies
# Correction of surface distortion not needed here since rho * A * z / A = rho * z.
function anom!(x, scale, now, ref)
    @. x = scale * (now - ref)
    return nothing
end

function columnanom_mantle!(sim::Simulation)
    anom!(sim.now.columnanoms.mantle, sim.p.rho_uppermantle, sim.now.u, sim.ref.u)
    return nothing
end

function columnanom_litho!(sim::Simulation)
    anom!(sim.now.columnanoms.litho, sim.p.rho_litho, sim.now.ue, sim.ref.ue)
    return nothing
end

function columnanom_ice!(sim::Simulation)
    anom!(sim.now.columnanoms.ice, sim.c.rho_ice, sim.now.H_af, sim.ref.H_af)
    return nothing
end

function columnanom_water!(sim::Simulation, ol::InteractiveOceanLoad)
    watercolumn!(sim)
    anom!(sim.now.columnanoms.seawater, sim.c.rho_seawater, sim.now.z_ss, sim.ref.z_ss)
    sim.now.columnanoms.seawater .*= sim.now.maskocean .* sim.p.maskactive
    return nothing
end

function columnanom_water!(sim::Simulation, ol::NoOceanLoad)
    watercolumn!(sim)
    sim.now.columnanoms.seawater .= 0
    return nothing
end

function watercolumn!(sim::Simulation)
    watercolumn!(sim.now.H_water, sim.now.H_ice, sim.now.maskgrounded, sim.now.z_b,
        sim.now.z_ss, sim.c, sim.tools.prealloc.buffer_x)
    return nothing
end

function watercolumn!(H_water, H_ice, maskgrounded, z_b, z_ss, c, buffer)
    # water column height in absence of ice
    buffer .= max.(z_ss .- z_b, 0)

    # if ice thickness lesser than threshold, only impose water column
    # if ice thickness greater than threshold, impose difference (accounting for floatation)
    H_water .= (H_ice .<= 1) .* buffer .+
        not.(maskgrounded) .* (H_ice .> 1) .*
        (buffer .- (H_ice .* (c.rho_ice / c.rho_seawater)))
    return nothing
end

function watercolumn(H_ice, maskgrounded, z_b, z_ss, c)
    H_water, buffer = similar(H_ice), similar(H_ice)
    watercolumn!(H_water, H_ice, maskgrounded, z_b, z_ss, c, buffer)
    return H_water
end

function columnanom_sediment!(sim::Simulation)
end

function columnanom_load!(sim::Simulation)
    canoms = sim.now.columnanoms
    @. canoms.load .= sim.p.maskactive * (canoms.ice + canoms.seawater + canoms.sediment)
    return nothing
end

function columnanom_full!(sim::Simulation)
    canoms = sim.now.columnanoms
    @. canoms.full = canoms.load + sim.p.maskactive * (canoms.litho + canoms.mantle)
    return nothing
end

function mass_anom(sim::Simulation)
    return sim.domain.A .* (sim.now.columnanoms.full) # .-
        # sim.c.rho_seawater .* sim.now.z_bsl .* sim.now.maskocean .* sim.ref.maskactive
end

function mass_anom(A, canom_full)
    return A * canom_full
end