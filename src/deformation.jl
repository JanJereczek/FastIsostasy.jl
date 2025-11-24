#####################################################
# Mantle response
#####################################################

"""
$(TYPEDSIGNATURES)

Update the time derivative of the viscous displacement `dudt` based on an [`Model`](@ref):
- `RigidMantle`: no deformation, `dudt` is zero.
- `RelaxedMantle` with `LaterallyConstantLithosphere`: uses ELRA (LeMeur & Huybrechts 1996)
  to compute the viscous response. This also works with laterally-variable relaxation time,
  as proposed in Van Calcar et al., in rev.
- `RelaxedMantle` with `LaterallyVariableLithosphere`: not implemented. This corresponds to
  what is described in Coulon et al. (2021) but is not yet implemented.
- `MaxwellMantle` with `LaterallyConstantLithosphere` or `RigidLithosphere`: not implemented.
  This corresponds to what is described in Bueler et al. (2007) but is not yet implemented.
- `MaxwellMantle` with `LaterallyVariableLithosphere`: This corresponds to the approach
  of Swierczek-Jereczek et al. (2024).
"""
function update_dudt!(dudt, u, sim, t, earth::SolidEarth)
    update_dudt!(dudt, u, sim, t, earth.mantle, earth.lithosphere)
end

function update_dudt!(dudt, u, sim, t, mantle::RigidMantle, litho)
    dudt .= 0
    return nothing
end

function update_dudt!(dudt, u, sim, t, mantle::RelaxedMantle, 
    litho::L) where {L<:AbstractLithosphere}
    
    update_deformation_rhs!(sim, u)

    @. sim.tools.prealloc.buffer_x = - (sim.now.columnanoms.load +
        sim.now.columnanoms.litho) * sim.c.g * sim.domain.K ^ 2
    
    samesize_conv!(sim.now.u_eq, sim.tools.prealloc.buffer_x,
        sim.tools.viscous_convo, sim.tools.conv_helpers,
        sim.domain, sim.bcs.viscous_displacement,
        sim.bcs.viscous_displacement.space)

    @. dudt = 1 / sim.solidearth.tau * (sim.now.u_eq - sim.now.u)
    return nothing
    
end

function update_dudt!(dudt, u, sim, t, mantle::RelaxedMantle,
    litho::LaterallyVariableLithosphere)
    error("Relaxed rheology is not implemented for laterally variable lithosphere.")
end

function update_dudt!(dudt, u, sim, t, mantle::MaxwellMantle,
    litho::L) where {L<:AbstractLithosphere}
    # error("Viscous rheology is not implemented for laterally constant lithosphere.")

    tools = sim.tools
    P = tools.prealloc
    dt = sim.opts.diffeq.dt_min * sim.c.seconds_per_year

    # helper variables
    nabla = P.buffer_xx
    @. nabla = 2 * sim.solidearth.effective_viscosity * sim.domain.pseudodiff *
        sim.solidearth.pseudodiff_scaling

    beta = P.buffer_x
    @. beta = sim.solidearth.rho_uppermantle * sim.c.g + sim.solidearth.litho_rigidity *
        sim.domain.pseudodiff ^ 4

    # fourier transform load
    @. P.fftF = - (sim.now.columnanoms.load +
        sim.now.columnanoms.litho) * sim.c.g * sim.domain.K ^ 2
    tools.pfft! * P.fftF

    # fourier transform u
    P.fftU .= u
    tools.pfft! * P.fftU

    # compute the right-hand side of the deformation equation
    @. P.fftrhs = ((nabla - (dt/2)*beta) * P.fftU + dt * P.fftF) / (nabla + (dt/2)*beta)
    tools.pifft! * P.fftrhs

    P.rhs .= real.(P.fftrhs)
    apply_bc!(P.rhs, sim.bcs.viscous_displacement)
    u .= P.rhs
    sim.now.u .= u

    # dudt .= (P.rhs .- sim.now.u) ./ dt .* sim.c.seconds_per_year
    return nothing
end

function update_dudt!(dudt, u, sim, t, mantle::MaxwellMantle,
    lithosphere::LaterallyVariableLithosphere)
    domain, P = sim.domain, sim.tools.prealloc
    update_deformation_rhs!(sim, u)
    @. P.fftrhs = P.rhs * domain.K / (2 * sim.solidearth.effective_viscosity)
    sim.tools.pfft! * P.fftrhs
    @. P.fftrhs *= sim.solidearth.scaled_pseudodiff_inv
    sim.tools.pifft! * P.fftrhs
    dudt .= real.(P.fftrhs)
    dudt .*= sim.c.seconds_per_year
    apply_bc!(dudt, sim.bcs.viscous_displacement)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the right-hand side of the deformation equation.
"""
function update_deformation_rhs!(sim::Simulation, u)

    domain, P = sim.domain, sim.tools.prealloc
    @. P.rhs = -sim.c.g * sim.now.columnanoms.full
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x,
        P.buffer_xy, u, domain)

    @. P.Mxx = -sim.solidearth.litho_rigidity *
        muladd(sim.solidearth.litho_poissonratio, P.buffer_yy, P.buffer_xx)
    @. P.Myy = -sim.solidearth.litho_rigidity *
        muladd(sim.solidearth.litho_poissonratio, P.buffer_xx, P.buffer_yy)
    @. P.Mxy = -sim.solidearth.litho_rigidity *
        (1 - sim.solidearth.litho_poissonratio) * P.buffer_xy
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x, P.buffer_xy,
        P.Mxx, P.Myy, P.Mxy, domain)
    @. P.rhs += P.buffer_xx + muladd(2, P.buffer_xy, P.buffer_yy)

    P.buffer_x .= P.rhs
    samesize_conv!(P.rhs, P.buffer_x, sim.tools.smooth_convo,
        sim.tools.conv_helpers, sim.domain)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute the horizontal displacement field from the vertical displacement field `u`.
Equation used for this can be found at [https://en.wikipedia.org/wiki/Plate_theory].
Since we assume an isotropic material under pure bending, the in-plane displacement is 0.
The mid-surface of the thin plate is assumed to be at `litho_thickness / 2`.
"""
function thinplate_horizontal_displacement(u, litho_thickness, domain)
    u_x = zeros(domain)
    u_y = zeros(domain)
    thinplate_horizontal_displacement!(u_x, u_y, u, litho_thickness, domain)
    return u_x, u_y
end

function thinplate_horizontal_displacement!(u_x::M, u_y::M, u::M,
    litho_thickness::M, domain) where {M<:Matrix}
    dx!(u_x, u, domain)
    dy!(u_y, u, domain)
    @. u_x *= -litho_thickness / 2
    @. u_y *= -litho_thickness / 2
    return nothing
end

function thinplate_horizontal_displacement!(u_x::M, u_y::M, u::M,
    litho_thickness::M, domain) where {T<:AbstractFloat, M<:CuMatrix{T}}
    dx!(u_x, u, domain.Dx, domain.nx, domain.ny)
    dy!(u_y, u, domain.Dx, domain.nx, domain.ny)
    @. u_x *= -litho_thickness / 2  # T(30e3)
    @. u_y *= -litho_thickness / 2  # T(30e3)
    return nothing
end

#####################################################
# Lithosphere response
#####################################################
"""
$(TYPEDSIGNATURES)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from [^Farrell1972], see [GIATools](@ref).
"""
function update_elasticresponse!(sim::Simulation, lithosphere::L) where {L<:AbstractLithosphere}

    @. sim.tools.prealloc.buffer_x = sim.now.columnanoms.load * sim.domain.K ^ 2
    samesize_conv!(sim.now.ue, sim.tools.prealloc.buffer_x,
        sim.tools.elastic_convo, sim.tools.conv_helpers, sim.domain,
        sim.bcs.elastic_displacement, sim.bcs.elastic_displacement.space)
    # sim.now.ue .= samesize_conv(sim.now.columnanoms.load .* sim.domain.K .^ 2,
    #     sim.tools.elastic_convo, sim.domain)
    return nothing
end

function update_elasticresponse!(sim::Simulation, lithosphere::RigidLithosphere)
    sim.now.ue .= 0
    return nothing
end