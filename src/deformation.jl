#####################################################
# Mantle response
#####################################################

"""
$(TYPEDSIGNATURES)

Update the time derivative of the viscous displacement based on a dispatch is along
three mainly orthogonal axes: the mantle rheology, the lithosphere, and the FFT backend.

Main supported combinations are:
- [`RigidMantle`](@ref): no deformation, `dudt` is zero.
- [`RelaxedMantle`](@ref) with [`LaterallyConstantLithosphere`](@ref): uses ELRA [le_meur_comparison_1996](@citet)
  to compute the viscous response. This also works with laterally-variable relaxation time,
  as proposed by [coulon_contrasting_2021](@citet) and by [van_calcar_approximating_2026](@citet).
- [`RelaxedMantle`](@ref) with [`LaterallyVariableLithosphere`](@ref): not implemented. This corresponds to
  what is described by [coulon_contrasting_2021](@citet) but is not yet implemented.
- [`ViscousMantle`](@ref) with [`LaterallyConstantLithosphere`](@ref) or [`RigidLithosphere`](@ref): not implemented.
  This corresponds to what is described by [bueler_fast_2007](@citet) but is not yet implemented.
- [`ViscousMantle`](@ref) with [`LaterallyVariableLithosphere`](@ref): This corresponds to the approach
  of [swierczek-jereczek_fastisostasy_2024](@citet).
"""
function update_dudt!(dudt, u, sim, t, earth::SolidEarth)
    update_dudt!(dudt, u, sim, t, earth.mantle, earth.lithosphere, sim.opts.fft)
end

function update_dudt!(dudt, u, sim, t, mantle::RigidMantle, litho, fft)
    dudt .= 0
    return nothing
end

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::RelaxedMantle,
    litho::L,
    fft,
) where {L<:AbstractLithosphere}

    update_deformation_rhs!(sim, u)

    @. sim.tools.prealloc.buffer_x =
        - (sim.now.columnanoms.load + sim.now.columnanoms.litho) *
        sim.c.g *
        sim.domain.K ^ 2

    samesize_conv!(
        sim.now.u_eq,
        sim.tools.prealloc.buffer_x,
        sim.tools.viscous_convo,
        sim.tools.conv_helpers,
        sim.domain,
        sim.bcs.viscous_displacement,
        sim.bcs.viscous_displacement.space,
    )

    @. dudt = 1 / sim.solidearth.tau * (sim.now.u_eq - sim.now.u)
    return nothing

end

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::RelaxedMantle,
    litho::LaterallyVariableLithosphere,
    fft,
)
    error("Relaxed rheology is not implemented for laterally variable lithosphere.")
end

# =============================================================================
# TransientCreepMantle: steady Maxwell dashpot + N Kelvin-Voigt branches.
#
# Per Fourier mode (roadmap burgers.md §2.3), with k the wavenumber, all branches
# in series so they carry the same stress and their displacements add
# (`u = u_M + Σⱼ u_K[j]`):
#
#     A du_M/dt = F − β (u_M + Σⱼ u_K[j])                     A  = 2 η₁ k
#     Bⱼ du_K[j]/dt = F − β (u_M + Σⱼ u_K[j]) − Cⱼ u_K[j]     Cⱼ = 2 μ₂ⱼ k, Bⱼ = τⱼ Cⱼ
#
# Setting u_K ≡ 0 recovers the `ViscousMantle` equation exactly, which is what
# makes the Δ → 0 limit a machine-precision regression test.
#
# Explicit stepping is not an option: retardation times τⱼ can be decades while
# the Maxwell time is millennia, so the system is stiff. Crank–Nicolson on the
# coupled (N+1)-field system is A-stable. Writing dᵢ, eᵢ for the per-branch pair
# (A, 0) at i = 0 (Maxwell) and (Bⱼ, Cⱼ) at i = j (Kelvin branch j), every row has
# the same shape dᵢ duᵢ/dt = F − β·Σₖuₖ − eᵢ uᵢ: branch i couples to every other
# branch *only* through their common sum Σₖuₖ, never directly to a specific other
# branch. That is a rank-1 (not dense, not merely arrowhead) coupling in
# (u_M, u_K[1], …, u_K[N]), and CN turns it into
#
#     gᵢ uᵢⁿ⁺¹ + aβ Sⁿ⁺¹ = rᵢ,   gᵢ = dᵢ + a eᵢ,   S = Σₖ uₖ,   a = dt/2
#     rᵢ = (dᵢ − a eᵢ) uᵢⁿ − aβ Sⁿ + dt F           (F frozen over the half-step)
#
# Summing the first line over i and solving for Sⁿ⁺¹ (Sherman–Morrison for a
# diagonal-plus-rank-1 system) gives an O(N) closed form, no per-mode matrix
# factorisation:
#
#     Sⁿ⁺¹ = [ Σᵢ rᵢ/gᵢ ] / [ 1 + aβ Σᵢ 1/gᵢ ],   uᵢⁿ⁺¹ = (rᵢ − aβ Sⁿ⁺¹) / gᵢ
#
# For i = 0, (d₀ − a e₀)/g₀ = A/A = 1 exactly, so the Maxwell branch's own r₀/g₀
# term is just u_Mⁿ — and u_Mⁿ = Sⁿ − Σⱼ u_K[j]ⁿ (linearity), so the Maxwell branch
# never has to be materialised: only Sⁿ (= FFT of the state array `u`, which
# already *is* u_M + Σⱼ u_K[j]) and the Kelvin branches' own state are needed. See
# `update_dudt!(..., ::TransientCreepMantle{MT,N}, ...)` below for the expansion;
# setting N = 1 there reproduces the closed-form 2×2 solve this comment used to
# describe line for line.
# =============================================================================

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::TransientCreepMantle{MT,N},
    litho::Union{LaterallyConstantLithosphere,RigidLithosphere},
    fft::ComplexFFTBackend,
) where {MT,N}

    tools = sim.tools
    P = tools.prealloc
    domain, se = sim.domain, sim.solidearth
    dt = fixed_dt(sim.opts.integ) * sim.c.seconds_per_year
    a = dt / 2

    # `update_dudt!` is a *pure* RHS: the stepper calls it more than once per step
    # (init_problem!, FSAL priming, then once per accepted step), so it must not
    # advance state on every call. `u_K` is therefore held at the time `t_K` it is
    # valid for, and the result of the step is parked in `u_K_next`; the commit
    # happens on the first call at a genuinely later time. Two calls at the same
    # `t` then produce identical output, exactly like the ViscousMantle method.
    if t > sim.now.t_K
        sim.now.u_K .= sim.now.u_K_next
        sim.now.t_K = t
    end

    # Spectral coefficient fields shared by every branch (A = d₀, β as usual).
    A = P.buffer_xx
    @. A = 2 * se.effective_viscosity * domain.pseudodiff * se.pseudodiff_scaling

    beta = P.buffer_x
    @. beta = se.rho_uppermantle * sim.c.g + se.litho_rigidity * domain.pseudodiff ^ 4

    # --- to the spectrum: forward-transform every branch's current state --------
    # `fftrhs` is the real->complex staging buffer throughout, exactly as in the
    # N = 1 method it generalises.
    for j in 1:N
        u_Kj = view(sim.now.u_K, :, :, j)
        @. P.fftrhs = u_Kj
        mul!(view(P.fftK, :, :, j), tools.pfft!, P.fftrhs)
    end

    @. P.fftrhs =
        - (sim.now.columnanoms.load + sim.now.columnanoms.litho) *
        sim.c.g *
        domain.K ^ 2
    mul!(P.fftF, tools.pfft!, P.fftrhs)

    @. P.fftrhs = u
    mul!(P.fftU, tools.pfft!, P.fftrhs)        # Sⁿ = FFT(u_M + Σⱼ u_K[j])ⁿ

    # `common = dt F − aβ Sⁿ` is shared by every branch's rᵢ (see comment above).
    @. P.fftrhs = dt * P.fftF - a * beta * P.fftU

    # --- Σᵢ rᵢ/gᵢ and Σᵢ 1/gᵢ, elementwise over the spectrum --------------------
    # i = 0 (Maxwell) contributes u_Mⁿ = Sⁿ − Σⱼ u_K[j]ⁿ to the first sum and 1/A
    # to the second; both are folded in once, after the loop, instead of inside it
    # (u_Mⁿ is never materialised on its own — see the derivation above). Per
    # branch j, (dⱼ − a eⱼ)/gⱼ = (τⱼ − a)/(τⱼ + a): μ₂ⱼ cancels between numerator
    # and denominator (dⱼ = τⱼCⱼ, eⱼ = Cⱼ), so the −Σⱼ u_K[j]ⁿ · (dⱼ−aeⱼ)/gⱼ term
    # collapses to a *scalar*-weighted sum; only 1/gⱼ needs the spectral field Cⱼ.
    @. P.buffer_xy = 1 / A                     # Σᵢ 1/gᵢ, i = 0 term
    P.fftF .= 0                                 # Σᵢ rᵢ/gᵢ accumulator (F̂ baked into `common` already)
    for j in 1:N
        tau_j = mantle.kelvin_time[j] * sim.c.seconds_per_year
        mu2_j = mantle.shearmodulus / mantle.relaxation_strength[j]
        fftKj = view(P.fftK, :, :, j)
        @. P.buffer_yy =                       # gⱼ = Cⱼ (τⱼ + a)
            2 * mu2_j * domain.pseudodiff * se.pseudodiff_scaling * (tau_j + a)
        @. P.buffer_xy += 1 / P.buffer_yy
        @. P.fftF -= (2 * a / (tau_j + a)) * fftKj
    end
    @. P.fftF += P.fftU + P.fftrhs * P.buffer_xy
    @. P.fftF = P.fftF / (1 + a * beta * P.buffer_xy)   # Sⁿ⁺¹ = total viscous displacement

    # --- finalise each Kelvin branch: uⱼⁿ⁺¹ = ratioⱼ uⱼⁿ + (common − aβ Sⁿ⁺¹)/gⱼ -
    # (each entry depends only on itself, so the in-place update is safe, exactly
    # like the N = 1 method's self-referential `P.fftK` update.)
    for j in 1:N
        tau_j = mantle.kelvin_time[j] * sim.c.seconds_per_year
        mu2_j = mantle.shearmodulus / mantle.relaxation_strength[j]
        fftKj = view(P.fftK, :, :, j)
        @. P.buffer_yy =
            2 * mu2_j * domain.pseudodiff * se.pseudodiff_scaling * (tau_j + a)
        ratio_j = (tau_j - a) / (tau_j + a)
        @. fftKj = ratio_j * fftKj + (P.fftrhs - a * beta * P.fftF) / P.buffer_yy
    end

    # --- back to real space ------------------------------------------------------
    mul!(P.fftU, tools.pifft!, P.fftF)
    P.rhs .= real.(P.fftU)
    apply_bc!(P.rhs, sim.bcs.viscous_displacement)
    # As a rate, so the stepper reproduces u^{n+1} exactly — see the ViscousMantle
    # method for why this must not write `u` directly.
    @. dudt = (P.rhs - u) / dt * sim.c.seconds_per_year

    for j in 1:N
        fftKj = view(P.fftK, :, :, j)
        mul!(P.fftU, tools.pifft!, fftKj)
        P.rhs .= real.(P.fftU)
        # The same (linear) BC on every branch keeps u = u_M + Σⱼ u_K[j] exact.
        apply_bc!(P.rhs, sim.bcs.viscous_displacement)
        view(sim.now.u_K_next, :, :, j) .= P.rhs
    end

    return nothing
end

# --- guard rails: combinations the semi-implicit solve does not cover yet -----
#
# These must stay *disjoint* from the real method above and from each other on at
# least one argument (here: `litho`, or `fft`), not merely dominated by it. A
# generic `(mantle::TransientCreepMantle, litho, fft)` catch-all used to work when
# the real method pinned N to the literal `1` (`TransientCreepMantle{MT,1}`),
# which made it a strict subtype of the catch-all's unconstrained
# `TransientCreepMantle` on that argument. Once N became free (`{MT,N} where
# {MT,N}`) it is *equal* to the catch-all's constraint on `mantle`, not a strict
# subtype, and Julia's method-specificity check does not resolve that tie just
# because the other arguments (`litho`, `fft`) are strictly narrower — it reports
# the pair as ambiguous instead of picking the narrower one. Splitting the
# catch-all into the two litho/fft combinations it actually needs to cover keeps
# every pair of methods below disjoint on at least one argument, so no tie is
# ever reached regardless of how `mantle` is constrained.

update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::TransientCreepMantle,
    litho::Union{LaterallyConstantLithosphere,RigidLithosphere},
    fft::RealFFTBackend,
) = error(
    "TransientCreepMantle does not yet support RealFFTBackend (roadmap " *
    "burgers.md §4: 'implement against ComplexFFTBackend first'). Use " *
    "SolverOptions(fft = ComplexFFTBackend()).",
)

update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::TransientCreepMantle,
    litho::LaterallyVariableLithosphere,
    fft,
) = error(
    "TransientCreepMantle does not support LaterallyVariableLithosphere: the " *
    "v1 effective-viscosity trick has no proven analogue for the coupled " *
    "(N+1)-field system (roadmaps/burgers.md §8). Use " *
    "LaterallyConstantLithosphere or RigidLithosphere.",
)

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::ViscousMantle,
    litho::L,
    fft::ComplexFFTBackend,
) where {L<:AbstractLithosphere}

    tools = sim.tools
    P = tools.prealloc
    dt = fixed_dt(sim.opts.integ) * sim.c.seconds_per_year

    # helper variables
    nabla = P.buffer_xx
    @. nabla =
        2 *
        sim.solidearth.effective_viscosity *
        sim.domain.pseudodiff *
        sim.solidearth.pseudodiff_scaling

    beta = P.buffer_x
    @. beta =
        sim.solidearth.rho_uppermantle * sim.c.g +
        sim.solidearth.litho_rigidity * sim.domain.pseudodiff ^ 4

    # Out-of-place plans (mul!) preserve their inputs, so stage each real field into
    # P.fftrhs and transform it into a distinct buffer.
    # fourier transform load -> P.fftF
    @. P.fftrhs =
        - (sim.now.columnanoms.load + sim.now.columnanoms.litho) *
        sim.c.g *
        sim.domain.K ^ 2
    mul!(P.fftF, tools.pfft!, P.fftrhs)

    # fourier transform u -> P.fftU
    @. P.fftrhs = u
    mul!(P.fftU, tools.pfft!, P.fftrhs)

    # compute the right-hand side of the deformation equation, then inverse-transform
    @. P.fftrhs = ((nabla - (dt/2)*beta) * P.fftU + dt * P.fftF) / (nabla + (dt/2)*beta)
    mul!(P.fftF, tools.pifft!, P.fftrhs)

    P.rhs .= real.(P.fftF)
    apply_bc!(P.rhs, sim.bcs.viscous_displacement)

    # The Crank-Nicolson step above already produced u^{n+1}. Hand it back as a
    # *rate*, so the stepper's `u + Δt·dudt` reproduces it exactly (`dt` is in
    # seconds, the stepper's Δt in years).
    #
    # Writing `u .= P.rhs` here instead — mutating the integrator's own state from
    # inside the RHS and leaving `dudt` untouched — is what produced the NaN
    # tracked in roadmaps/ad_inversion.md §8: `dudt` is `integ.ks[1]`, a `similar`
    # array this method never wrote, so the stepper added `Δt ·` uninitialised
    # memory on top of the already-updated `u`. As a rate the method is also pure,
    # which is what lets it be called more than once per step (init_problem!, FSAL
    # priming) without advancing anything twice.
    @. dudt = (P.rhs - u) / dt * sim.c.seconds_per_year
    return nothing
end

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::ViscousMantle,
    litho::L,
    fft::RealFFTBackend,
) where {L<:AbstractLithosphere}

    tools = sim.tools
    P = tools.prealloc
    domain = sim.domain
    dt = fixed_dt(sim.opts.integ) * sim.c.seconds_per_year
    nx2 = domain.nx ÷ 2 + 1

    # frequency-domain coefficient arrays: borrow the first nx2 rows of real buffers.
    # @. is NOT used here because it would turn view(...) into view.(...) (broadcasted view).
    nabla = view(P.buffer_xx, 1:nx2, :)
    nabla .=
        2 .* view(sim.solidearth.effective_viscosity, 1:nx2, :) .*
        view(domain.pseudodiff, 1:nx2, :) .*
        view(sim.solidearth.pseudodiff_scaling, 1:nx2, :)

    beta = view(P.buffer_x, 1:nx2, :)
    beta .=
        sim.solidearth.rho_uppermantle .* sim.c.g .+
        view(sim.solidearth.litho_rigidity, 1:nx2, :) .*
        view(domain.pseudodiff, 1:nx2, :) .^ 4

    # rfft of load — stage into P.rhs (real), then transform out-of-place into P.fftF
    @. P.rhs =
        -(sim.now.columnanoms.load + sim.now.columnanoms.litho) * sim.c.g * domain.K ^ 2
    mul!(P.fftF, tools.pfft!, P.rhs)

    # rfft of u
    mul!(P.fftU, tools.pfft!, u)

    # frequency-domain update
    @. P.fftrhs = ((nabla - (dt/2)*beta) * P.fftU + dt * P.fftF) / (nabla + (dt/2)*beta)

    # irfft → result is directly real, no real.(.) needed
    mul!(P.rhs, tools.pifft!, P.fftrhs)

    apply_bc!(P.rhs, sim.bcs.viscous_displacement)
    u .= P.rhs
    sim.now.u .= u
    return nothing
end

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::ViscousMantle,
    lithosphere::LaterallyVariableLithosphere,
    fft::RealFFTBackend,
)
    domain, P = sim.domain, sim.tools.prealloc
    nx2 = domain.nx ÷ 2 + 1
    update_deformation_rhs!(sim, u)
    # stage the real-valued input in P.buffer_yy (P.fftrhs is now complex/half-sized)
    @. P.buffer_yy = P.rhs * domain.K / (2 * sim.solidearth.effective_viscosity)
    mul!(P.fftrhs, sim.tools.pfft!, P.buffer_yy)
    P.fftrhs .*= view(sim.solidearth.scaled_pseudodiff_inv, 1:nx2, :)
    mul!(P.rhs, sim.tools.pifft!, P.fftrhs)
    dudt .= P.rhs
    dudt .*= sim.c.seconds_per_year
    apply_bc!(dudt, sim.bcs.viscous_displacement)
    return nothing
end

function update_dudt!(
    dudt,
    u,
    sim,
    t,
    mantle::ViscousMantle,
    lithosphere::LaterallyVariableLithosphere,
    fft::ComplexFFTBackend,
)
    domain, P = sim.domain, sim.tools.prealloc
    update_deformation_rhs!(sim, u)
    # Stage the real-valued rhs into a complex buffer, then apply the out-of-place
    # plans with `mul!` (dest ≠ src) so each transform's input is preserved.
    @. P.fftU = P.rhs * domain.K / (2 * sim.solidearth.effective_viscosity)
    mul!(P.fftrhs, sim.tools.pfft!, P.fftU)
    @. P.fftrhs *= sim.solidearth.scaled_pseudodiff_inv
    mul!(P.fftU, sim.tools.pifft!, P.fftrhs)
    dudt .= real.(P.fftU)
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
    update_second_derivatives!(
        P.buffer_xx,
        P.buffer_yy,
        P.buffer_x,
        P.buffer_xy,
        u,
        domain,
    )

    @. P.Mxx =
        -sim.solidearth.litho_rigidity *
        muladd(sim.solidearth.litho_poissonratio, P.buffer_yy, P.buffer_xx)
    @. P.Myy =
        -sim.solidearth.litho_rigidity *
        muladd(sim.solidearth.litho_poissonratio, P.buffer_xx, P.buffer_yy)
    @. P.Mxy =
        -sim.solidearth.litho_rigidity *
        (1 - sim.solidearth.litho_poissonratio) *
        P.buffer_xy
    update_second_derivatives!(
        P.buffer_xx,
        P.buffer_yy,
        P.buffer_x,
        P.buffer_xy,
        P.Mxx,
        P.Myy,
        P.Mxy,
        domain,
    )
    @. P.rhs += P.buffer_xx + muladd(2, P.buffer_xy, P.buffer_yy)

    P.buffer_x .= P.rhs
    samesize_conv!(
        P.rhs,
        P.buffer_x,
        sim.tools.smooth_convo,
        sim.tools.conv_helpers,
        sim.domain,
    )
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute the horizontal displacement field from the vertical displacement field `u`.
Equations can be found at [https://en.wikipedia.org/wiki/Plate_theory].
Since we assume an isotropic material under pure bending, the in-plane displacement is 0.
The mid-surface of the thin plate is assumed to be at `litho_thickness / 2`.
"""
function thinplate_horizontal_displacement(u, litho_thickness, domain)
    u_x = zeros(domain)
    u_y = zeros(domain)
    thinplate_horizontal_displacement!(u_x, u_y, u, litho_thickness, domain)
    return u_x, u_y
end

function thinplate_horizontal_displacement!(
    u_x::M,
    u_y::M,
    u::M,
    litho_thickness::M,
    domain,
) where {M<:AbstractMatrix}
    dx!(u_x, u, domain)
    dy!(u_y, u, domain)
    @. u_x *= -litho_thickness / 2
    @. u_y *= -litho_thickness / 2
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
function update_elasticresponse!(
    sim::Simulation,
    lithosphere::L,
) where {L<:AbstractLithosphere}

    @. sim.tools.prealloc.buffer_x = sim.now.columnanoms.load * sim.domain.K ^ 2
    samesize_conv!(
        sim.now.ue,
        sim.tools.prealloc.buffer_x,
        sim.tools.elastic_convo,
        sim.tools.conv_helpers,
        sim.domain,
        sim.bcs.elastic_displacement,
        sim.bcs.elastic_displacement.space,
    )
    # sim.now.ue .= samesize_conv(sim.now.columnanoms.load .* sim.domain.K .^ 2,
    #     sim.tools.elastic_convo, sim.domain)
    return nothing
end

function update_elasticresponse!(sim::Simulation, lithosphere::RigidLithosphere)
    sim.now.ue .= 0
    return nothing
end