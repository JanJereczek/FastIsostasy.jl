"""
    update_diagnostics!(dudt, u, fip, t)

Update all the diagnotisc variables, i.e. all fields of `fip.now` apart
from the displacement, which requires an integrator.
"""
function update_diagnostics!(dudt, u, fip::FastIsoProblem, t)
    # CAUTION: Order really matters here!
    push!(fip.nout.t_steps_ode, t)                  # Add time step to output.
    apply_bc!(u, fip.bcs.u)                         # Make sure that u satisfies BC.
    apply_bc!(fip.now.H_ice, t, fip.bcs.h_ice)      # Apply ice thickness BC.
    update_Haf!(fip)
    update_loadcolumns!(fip, fip.bcs.z_ss)
    columnanom_load!(fip)

    # As integration requires smaller time steps than diagnostics,
    # only update diagnostics every fip.opts.dt_diagnostics or fip.ncout.t
    update_diagnostics = ((t / fip.opts.dt_diagnostics) >= fip.now.countupdates + 1)

    if update_diagnostics
        # @show t
        # if elastic update placed after dz_ss, worse match with (Spada et al. 2011)
        update_elasticresponse!(fip, fip.em.lithosphere)
        columnanom_litho!(fip)
        update_bsl!(fip)
        update_sealevel!(fip, fip.bcs.z_ss)
        fip.now.countupdates += 1
    end

    columnanom_full!(fip)
    update_dudt!(dudt, u, fip, t, fip.em)
    fip.now.dudt .= dudt
    columnanom_mantle!(fip)
    update_bedrock!(fip, u)
    return nothing
end

#####################################################
# Viscous response
#####################################################

"""
    update_dudt!(dudt, u, fip, t, model::SolidEarthModel)

Update the time derivative of the viscous displacement `dudt` based on an [`SolidEarthModel`](@ref):
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
function update_dudt!(dudt, u, fip, t, model::SolidEarthModel)
    update_dudt!(dudt, u, fip, t, model.rheology, model.lithosphere)
end

function update_dudt!(dudt, u, fip, t, mantle::RigidMantle, litho)
    dudt .= 0
    return nothing
end

function update_dudt!(dudt, u, fip, t, mantle::RelaxedMantle, 
    litho::LaterallyConstantLithosphere)
    
    update_deformation_rhs!(fip, u)

    @. fip.tools.prealloc.buffer_x = - (fip.now.columnanoms.load +
        fip.now.columnanoms.litho) * fip.c.g * fip.Omega.K ^ 2
    
    samesize_conv!(fip.now.u_eq, fip.tools.prealloc.buffer_x,
        fip.tools.viscous_convo, fip.Omega, fip.bcs.u, fip.bcs.u.space)

    @. dudt = 1 / fip.p.tau * (fip.now.u_eq - fip.now.u)
    return nothing
    
end

function update_dudt!(dudt, u, fip, t, mantle::RelaxedMantle,
    litho::LaterallyVariableLithosphere)
    error("Relaxed rheology is not implemented for laterally variable lithosphere.")
end

function update_dudt!(dudt, u, fip, t, mantle::MaxwellMantle,
    litho::L) where {L<:AbstractLithosphere}
    error("Viscous rheology is not implemented for laterally constant lithosphere.")
    # fft(load, t + dt/2)
    # U_now = fft(u_now)
    # U_next = (2 * eta * mu * kappa - dt/2*beta) * U_now +
    #     dt * load
    # U_next ./= (2 * eta * mu * kappa + dt/2*beta)
    # u_next = fftinv(U_next)
end

function update_dudt!(dudt, u, fip, t, mantle::MaxwellMantle,
    lithosphere::LaterallyVariableLithosphere)
    Omega, P = fip.Omega, fip.tools.prealloc
    update_deformation_rhs!(fip, u)
    @. P.fftrhs = P.rhs * Omega.K / (2 * fip.p.effective_viscosity)
    fip.tools.pfft! * P.fftrhs
    @. P.fftrhs *= Omega.pseudodiff_inv
    fip.tools.pifft! * P.fftrhs
    dudt .= real.(P.fftrhs)
    dudt .*= fip.c.seconds_per_year

    apply_bc!(dudt, fip.bcs.u)
    return nothing
end

"""
    update_deformation_rhs!(fip)

Update the right-hand side of the deformation equation.
"""
function update_deformation_rhs!(fip::FastIsoProblem, u)

    Omega, P = fip.Omega, fip.tools.prealloc
    @. P.rhs = -fip.c.g * fip.now.columnanoms.full
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x,
        P.buffer_xy, u, Omega)
    # @. P.Mxx = -fip.p.litho_rigidity * (P.buffer_xx +
    #     fip.p.litho_poissonratio * P.buffer_yy)
    # @. P.Myy = -fip.p.litho_rigidity * (P.buffer_yy +
    #     fip.p.litho_poissonratio * P.buffer_xx)
    # @. P.Mxy = -fip.p.litho_rigidity * (1 - fip.p.litho_poissonratio) * P.buffer_xy

    @. P.Mxx = -fip.p.litho_rigidity *
        muladd(fip.p.litho_poissonratio, P.buffer_yy, P.buffer_xx)
    @. P.Myy = -fip.p.litho_rigidity *
        muladd(fip.p.litho_poissonratio, P.buffer_xx, P.buffer_yy)
    @. P.Mxy = -fip.p.litho_rigidity * (1 - fip.p.litho_poissonratio) * P.buffer_xy
    update_second_derivatives!(P.buffer_xx, P.buffer_yy, P.buffer_x, P.buffer_xy,
        P.Mxx, P.Myy, P.Mxy, Omega)
    @. P.rhs += P.buffer_xx + muladd(2, P.buffer_xy, P.buffer_yy)
    return nothing
end

"""
    horizontal_displacement(u, litho_thickness, Omega)

Compute the horizontal displacement field from the vertical displacement field `u`.
Equation used for this can be found at [https://en.wikipedia.org/wiki/Plate_theory].
Since we assume an isotropic material under pure bending, the in-plane displacement is 0.
The mid-surface of the thin plate is assumed to be at `litho_thickness / 2`.
"""
function thinplate_horizontal_displacement(u, litho_thickness, Omega)
    u_x = null(Omega)
    u_y = null(Omega)
    thinplate_horizontal_displacement!(u_x, u_y, u, litho_thickness, Omega)
    return u_x, u_y
end

function thinplate_horizontal_displacement!(u_x::M, u_y::M, u::M,
    litho_thickness::M, Omega) where {M<:Matrix}
    dx!(u_x, u, Omega)
    dy!(u_y, u, Omega)
    @. u_x *= -litho_thickness / 2
    @. u_y *= -litho_thickness / 2
    return nothing
end

function thinplate_horizontal_displacement!(u_x::M, u_y::M, u::M,
    litho_thickness::M, Omega) where {T<:AbstractFloat, M<:CuMatrix{T}}
    dx!(u_x, u, Omega.Dx, Omega.nx, Omega.ny)
    dy!(u_y, u, Omega.Dx, Omega.nx, Omega.ny)
    @. u_x *= T(30e3)   #-litho_thickness / 2
    @. u_y *= T(30e3)   #-litho_thickness / 2
    return nothing
end

#####################################################
# Elastic response
#####################################################
"""
    update_elasticresponse!(fip::FastIsoProblem)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from [^Farrell1972], see [FastIsoTools](@ref).
"""
function update_elasticresponse!(fip::FastIsoProblem, lith::L) where {L<:AbstractLithosphere}

    @. fip.tools.prealloc.buffer_x = fip.now.columnanoms.load * fip.Omega.K ^ 2
    samesize_conv!(fip.now.ue, fip.tools.prealloc.buffer_x,
        fip.tools.elastic_convo, fip.Omega, fip.bcs.u_e, fip.bcs.u_e.space)
    # fip.now.ue .= samesize_conv(fip.now.columnanoms.load .* fip.Omega.K .^ 2,
    #     fip.tools.elastic_convo, fip.Omega)
    return nothing
end

function update_elasticresponse!(fip::FastIsoProblem, lith::RigidLithosphere)
    fip.now.ue .= 0
    return nothing
end

"""
    build_greenintegrand(distance::Vector{T}, 
        greenintegrand_coeffs::Vector{T}) where {T<:AbstractFloat}

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function build_greenintegrand(
    distance::Vector{T},
    greenintegrand_coeffs::Vector{T},
) where {T<:AbstractFloat}

    greenintegrand_interp = linear_interpolation(distance, greenintegrand_coeffs)
    compute_greenintegrand_entry_r(r::T) = get_loadgreen(
        r, distance, greenintegrand_coeffs, greenintegrand_interp)
    greenintegrand_function(x::T, y::T) = compute_greenintegrand_entry_r( get_r(x, y) )
    return greenintegrand_function
end

"""
    get_loadgreen(r::T, rm::Vector{T}, greenintegrand_coeffs::Vector{T},     
        interp_greenintegrand_::Interpolations.Extrapolation) where {T<:AbstractFloat}

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function get_loadgreen(
    r::T,
    rm::Vector{T},
    greenintegrand_coeffs::Vector{T},
    interp_greenintegrand_::Interpolations.Extrapolation,
) where {T<:AbstractFloat}

    if r < 0.01
        return greenintegrand_coeffs[1] / ( rm[2] * T(1e12) )
    elseif r > rm[end]
        return T(0.0)
    else
        return interp_greenintegrand_(r) / ( r * T(1e12) )
    end
end

"""
    get_elasticgreen(Omega, quad_support, quad_coeffs)

Integrate load response over field by using 2D quadrature with specified
support points and associated coefficients.
"""
function get_elasticgreen(
    Omega::RegionalComputationDomain{T, M},
    greenintegrand_function::Function,
    quad_support::Vector{T},
    quad_coeffs::Vector{T},
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    dx, dy = Omega.dx, Omega.dy
    elasticgreen = fill(T(0), Omega.nx, Omega.ny)

    @inbounds for i = 1:Omega.nx, j = 1:Omega.ny
        p = i - Omega.mx - 1
        q = j - Omega.my - 1
        elasticgreen[j, i] = quadrature2D(
            greenintegrand_function,
            quad_support,
            quad_coeffs,
            p*dx,
            (p+1)*dx,
            q*dy,
            (q+1)*dy,
        )
    end
    return elasticgreen
end