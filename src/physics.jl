struct PrecomputedTerms{T<:AbstractFloat}
    loadresponse::AbstractMatrix{T}
    fourier_loadresponse::AbstractMatrix{Complex{T}}
    fourier_left_term::AbstractMatrix{T}
    fourier_right_term::AbstractMatrix{T}
    forward_fft::FFTW.FFTWPlan
    inverse_fft::AbstractFFTs.ScaledPlan
end

"""

    precompute_terms(
        dt::T,
        Omega::ComputationDomain{T},
        p::SolidEarthParams{T},
        c::PhysicalConstants{T};
        quad_precision::Int = 4,
    ) where {T<:AbstractFloat}

Return a `struct` containing pre-computed tools to perform forward-stepping. Takes the
time step `dt`, the ComputationDomain `Omega`, the solid-Earth parameters `p` and 
physical constants `c` as input.
"""
@inline function precompute_terms(
    dt::T,
    Omega::ComputationDomain{T},
    p::SolidEarthParams{T},
    c::PhysicalConstants{T};
    quad_precision::Int = 4,
) where {T<:AbstractFloat}

    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    loadresponse = get_integrated_loadresponse(Omega, quad_support, quad_coeffs)
    fourier_left_term, fourier_right_term = get_cranknicolson_factors(
        Omega,
        dt,
        p,
        c,
    )
    p1, p2 = plan_twoway_fft(Omega.X)

    return PrecomputedTerms(
        loadresponse,
        p1 * loadresponse,
        fourier_left_term,
        fourier_right_term,
        p1,
        p2,
    )
end


"""

    function forward_isostasy!(
        Omega::ComputationDomain,
        t_vec::AbstractVector{T},
        u3D_elastic::Array{T, 3},
        u3D_viscous::Array{T, 3},
        sigma_zz::AbstractMatrix{T},
        tools::PrecomputedTerms,
        p::SolidEarthParams,
        c::PhysicalConstants,
    ) where {T<:AbstractFloat}

Integrates isostasy model over provided time vector.
"""
@inline function forward_isostasy!(
    Omega::ComputationDomain,
    t_vec::AbstractVector{T},       # the output time vector
    u3D_elastic::Array{T, 3},
    u3D_viscous::Array{T, 3},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    p::SolidEarthParams,
    c::PhysicalConstants;
    dt_refine=fill(100.0, length(t_vec)-1)::AbstractVector{T},   # refine time step for internal computation
) where {T}

    for i in eachindex(t_vec)[2:end]
        dt = t_vec[i] - t_vec[i-1]
        u3D_elastic[:, :, i], u3D_viscous[:, :, i] = forwardstep_isostasy(
            Omega,
            dt / dt_refine[i-1],
            dt,
            u3D_viscous[:, :, i-1],
            sigma_zz,
            tools,
            p,
            c,
        )
    end
end

"""

    forwardstep_isostasy(
        Omega::ComputationDomain,
        dt::T,
        u_viscous::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::PrecomputedTerms,
        p::SolidEarthParams,
        c::PhysicalConstants,
        viscous_solver="Euler"::String,
        dt_refine=100::Int,
    ) where {T<:AbstractFloat}

Forward-stepping of GIA model.
"""
@inline function forwardstep_isostasy(
    Omega::ComputationDomain,
    dt::T,
    dt_out::T,
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    p::SolidEarthParams,
    c::PhysicalConstants;
    viscous_solver="Euler"::String,
) where {T<:AbstractFloat}

    # load Ψ is defined as mass per surface area --> Ψ = - σ_zz / g
    # because σ_zz = rho_ice * g * H
    u_elastic_next = compute_elastic_response(Omega, tools, -sigma_zz ./ c.g )

    if viscous_solver == "Euler"

        t_internal = range(0, stop = dt_out, step = dt)
        u_internal = zeros(T, size(Omega.X)..., length(t_internal))
        u_internal[:, :, 1] = u_viscous
        for i in eachindex(t_internal)[2:end]
            u_internal[:, :, i] = apply_bc(euler_viscous_response(
                Omega,
                dt,
                u_internal[:, :, i-1],
                sigma_zz,
                tools,
                c,
                p,
            ))
        end
        u_viscous_next = u_internal[:, :, end]
    
    elseif viscous_solver == "CrankNicolson"
        u_viscous_next = apply_bc( cranknicolson_viscous_response(
            dt,
            u_viscous,
            sigma_zz,
            tools,
        ) )
    end

    return u_elastic_next, u_viscous_next
end

"""

    cranknicolson_viscous_response(
        dt::T,
        u_viscous::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::PrecomputedTerms,
    )

Return viscous response based on Crank-Nicolson time discretization.
Only valid for solid-Earth parameters that are constant over x, y.
"""
@inline function cranknicolson_viscous_response(
    dt::T,
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
) where {T<:AbstractFloat}
    # TODO FFT the load at t = tn + Δt / 2 giving ( hat_σ_zz )_pq
    num = tools.fourier_right_term .* (tools.forward_fft * u_viscous) + ( tools.forward_fft * (dt .* sigma_zz) )
    return real.(tools.inverse_fft * ( num ./ tools.fourier_left_term ))
end

"""

    euler_viscous_response(
        Omega::ComputationDomain,
        dt::T,
        u_current::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::PrecomputedTerms,
        c::PhysicalConstants,
        p::SolidEarthParams,
    )

Return viscous response based on explicit Euler time discretization.
Valid for solid-Earth parameters that can vary over x, y.
"""
@inline function euler_viscous_response(
    Omega::ComputationDomain,
    dt::T,
    u_current::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    c::PhysicalConstants,
    p::SolidEarthParams,
) where {T<:AbstractFloat}

    u_fourier_current = tools.forward_fft * u_current
    biharmonic_u = Omega.biharmonic_coeffs .* u_fourier_current
    loadgravity_term = sigma_zz - p.mantle_density .* c.g .* u_current
    rigidity_term = p.lithosphere_rigidity .* (tools.inverse_fft * biharmonic_u)
    lgr_term = loadgravity_term - rigidity_term

    eta_scaled = 2 .* p.halfspace_viscosity .* p.viscosity_scaling
    viscous_lgr = tools.forward_fft * ( lgr_term ./ eta_scaled )
    u_fourier_next = u_fourier_current + (dt ./ ( Omega.pseudodiff_coeffs .+ 1e-20 ) ) .* viscous_lgr
    return real.(tools.inverse_fft * u_fourier_next)
end

@inline function euler_viscous_response2(
    Omega::ComputationDomain,
    dt::T,
    u_current::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    c::PhysicalConstants,
    p::SolidEarthParams,
) where {T<:AbstractFloat}

    biharmonic_u = Omega.harmonic_coeffs .* ( tools.forward_fft * 
      ( p.lithosphere_rigidity .* (tools.inverse_fft *
      ( Omega.harmonic_coeffs .* (tools.forward_fft * u_current) ) ) ) )

    lgr_term = ( tools.forward_fft * sigma_zz -
      tools.forward_fft * (p.mantle_density .* c.g .* u_current) -
      biharmonic_u ) ./ ( Omega.pseudodiff_coeffs .* p.viscosity_scaling .+ 1e-20 )

    return u_current +
      dt ./ (2 .* p.halfspace_viscosity) .* real.(tools.inverse_fft * lgr_term)
end

"""

    apply_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at boundary is 0.
"""
@inline function apply_bc(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    return u .- T( ( sum(u[1,:]) + sum(u[:,1]) ) / sum(size(u)) )
end

"""

    get_cranknicolson_factors(
        Omega::ComputationDomain,
        dt::T,
        p::Union{LocalFields, SolidEarthParams},
        c::PhysicalConstants,
    )

Return two terms arising in the Crank-Nicolson scheme when applied to thepresent case.
"""
@inline function get_cranknicolson_factors(
    Omega::ComputationDomain,
    dt::T,
    p::SolidEarthParams,
    c::PhysicalConstants,
) where {T<:AbstractFloat}
    # mu already included in differential coeffs
    beta = ( p.mantle_density * c.g .+ p.lithosphere_rigidity .* Omega.biharmonic_coeffs )
    term1 = (2 .* p.halfspace_viscosity .* p.viscosity_scaling) .* Omega.pseudodiff_coeffs
    term2 = (dt/2) .* beta
    fourier_left_term = term1 + term2
    fourier_right_term = term1 - term2
    return fourier_left_term, fourier_right_term
end

"""

    plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}

Return forward-FFT and inverse-FFT plan to apply on array with same dimensions as `X`.
"""
@inline function plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    return plan_fft(X), plan_ifft(X)
end

"""

    compute_elastic_response(tools::PrecomputedTerms, load::AbstractMatrix{T})

Compute the elastic response of the solid Earth by convoluting the load with the
Green's function (elements obtained from Farell 1972). In the Fourier space, this
corresponds to a product which is subsequently transformed back into the time domain.
Use pre-computed integration tools to accelerate computation.
"""
@inline function compute_elastic_response(
    Omega::ComputationDomain,
    tools::PrecomputedTerms,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    if rem(Omega.N, 2) == 0
        return conv(load, tools.loadresponse)[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
    else
        return conv(load, tools.loadresponse)[Omega.N2+1:end-Omega.N2, Omega.N2+1:end-Omega.N2]
    end
end