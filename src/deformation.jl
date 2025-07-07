#####################################################
# Mantle response
#####################################################

"""
    update_dudt!(dudt, u, sim, t, model::Model)

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
function update_dudt!(dudt, u, sim, t, model::Model)
    update_dudt!(dudt, u, sim, t, model.mantle, model.lithosphere)
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
        sim.tools.viscous_convo, sim.domain, sim.bcs.viscous_displacement,
        sim.bcs.viscous_displacement.space)

    @. dudt = 1 / sim.p.tau * (sim.now.u_eq - sim.now.u)
    return nothing
    
end

function update_dudt!(dudt, u, sim, t, mantle::RelaxedMantle,
    litho::LaterallyVariableLithosphere)
    error("Relaxed rheology is not implemented for laterally variable lithosphere.")
end

function update_dudt!(dudt, u, sim, t, mantle::MaxwellMantle,
    litho::L) where {L<:AbstractLithosphere}
    error("Viscous rheology is not implemented for laterally constant lithosphere.")
    # fft(load, t + dt/2)
    # U_now = fft(u_now)
    # U_next = (2 * eta * mu * kappa - dt/2*beta) * U_now +
    #     dt * load
    # U_next ./= (2 * eta * mu * kappa + dt/2*beta)
    # u_next = fftinv(U_next)
end

function update_dudt!(dudt, u, sim, t, mantle::MaxwellMantle,
    lithosphere::LaterallyVariableLithosphere)
    domain, P = sim.domain, sim.tools.prealloc
    update_deformation_rhs!(sim, u)
    @. P.fftrhs = P.rhs * domain.K / (2 * sim.p.effective_viscosity)
    sim.tools.pfft! * P.fftrhs
    @. P.fftrhs *= domain.pseudodiff_inv
    sim.tools.pifft! * P.fftrhs
    dudt .= real.(P.fftrhs)
    dudt .*= sim.c.seconds_per_year
    apply_bc!(dudt, sim.bcs.viscous_displacement)
    return nothing
end


function update_dudt!(dudt, u, sim, t, mantle::MaxwellMantle,
    lithosphere::RigidLithosphere)
    domain, P = sim.domain, sim.tools.prealloc
    update_deformation_rhs!(sim, u)
    @. P.fftrhs = P.rhs * domain.K / (2 * sim.p.effective_viscosity)
    sim.tools.pfft! * P.fftrhs
    @. P.fftrhs *= domain.pseudodiff_inv
    sim.tools.pifft! * P.fftrhs
    dudt .= real.(P.fftrhs)
    dudt .*= sim.c.seconds_per_year
    apply_bc!(dudt, sim.bcs.viscous_displacement)
    return nothing
end

"""
    update_deformation_rhs!(sim)

Update the right-hand side of the deformation equation.
"""
function update_deformation_rhs!(sim::Simulation, u)

    domain, P = sim.domain, sim.tools.prealloc
    @. P.rhs = -sim.c.g * sim.now.columnanoms.full
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x,
        P.buffer_xy, u, domain)
    # @. P.Mxx = -sim.p.litho_rigidity * (P.buffer_xx +
    #     sim.p.litho_poissonratio * P.buffer_yy)
    # @. P.Myy = -sim.p.litho_rigidity * (P.buffer_yy +
    #     sim.p.litho_poissonratio * P.buffer_xx)
    # @. P.Mxy = -sim.p.litho_rigidity * (1 - sim.p.litho_poissonratio) * P.buffer_xy

    @. P.Mxx = -sim.p.litho_rigidity *
        muladd(sim.p.litho_poissonratio, P.buffer_yy, P.buffer_xx)
    @. P.Myy = -sim.p.litho_rigidity *
        muladd(sim.p.litho_poissonratio, P.buffer_xx, P.buffer_yy)
    @. P.Mxy = -sim.p.litho_rigidity * (1 - sim.p.litho_poissonratio) * P.buffer_xy
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x, P.buffer_xy,
        P.Mxx, P.Myy, P.Mxy, domain)
    @. P.rhs += P.buffer_xx + muladd(2, P.buffer_xy, P.buffer_yy)

    P.buffer_x .= P.rhs
    # samesize_conv!(P.rhs, P.buffer_x, sim.tools.smooth_convo, sim.domain)
    return nothing
end

"""
    horizontal_displacement(u, litho_thickness, domain)

Compute the horizontal displacement field from the vertical displacement field `u`.
Equation used for this can be found at [https://en.wikipedia.org/wiki/Plate_theory].
Since we assume an isotropic material under pure bending, the in-plane displacement is 0.
The mid-surface of the thin plate is assumed to be at `litho_thickness / 2`.
"""
function thinplate_horizontal_displacement(u, litho_thickness, domain)
    u_x = null(domain)
    u_y = null(domain)
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
    update_elasticresponse!(sim::Simulation)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from [^Farrell1972], see [GIATools](@ref).
"""
function update_elasticresponse!(sim::Simulation, lithosphere::L) where {L<:AbstractLithosphere}

    @. sim.tools.prealloc.buffer_x = sim.now.columnanoms.load * sim.domain.K ^ 2
    samesize_conv!(sim.now.ue, sim.tools.prealloc.buffer_x,
        sim.tools.elastic_convo, sim.domain, sim.bcs.elastic_displacement,
        sim.bcs.elastic_displacement.space)
    # sim.now.ue .= samesize_conv(sim.now.columnanoms.load .* sim.domain.K .^ 2,
    #     sim.tools.elastic_convo, sim.domain)
    return nothing
end

function update_elasticresponse!(sim::Simulation, lithosphere::RigidLithosphere)
    sim.now.ue .= 0
    return nothing
end