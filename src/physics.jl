struct PrecomputedTerms{T<:AbstractFloat}
    viscous_solver::String
    loadresponse::AbstractMatrix{T}
    fourier_loadresponse::AbstractMatrix{Complex{T}}
    fourier_left_term::AbstractMatrix{T}
    fourier_right_term::AbstractMatrix{T}
    forward_fft::AbstractFFTs.Plan
    inverse_fft::AbstractFFTs.ScaledPlan
end

"""

    precompute_terms(dt, Omega, p, c)

Return a `struct` containing pre-computed tools to perform forward-stepping. Takes the
time step `dt`, the ComputationDomain `Omega`, the solid-Earth parameters `p` and 
physical constants `c` as input.
"""
@inline function precompute_terms(
    dt::T,
    Omega::ComputationDomain{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T};
    quad_precision::Int = 4,
    viscous_solver::String = "Euler",
) where {T<:AbstractFloat}

    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    loadresponse = get_integrated_loadresponse(Omega, quad_support, quad_coeffs)

    if viscous_solver == "CrankNicolson"
        fourier_left_term, fourier_right_term = get_cranknicolson_factors(
            Omega,
            dt,
            p,
            c,
        )
    else
        fourier_left_term = zeros(T, Omega.N, Omega.N)
        fourier_right_term = copy(fourier_left_term)
    end

    if Omega.use_cuda
        Xgpu = CuArray(Omega.X)
        p1 = CUDA.CUFFT.plan_fft(Xgpu)
        p2 = CUDA.CUFFT.plan_ifft(Xgpu)
    else
        p1, p2 = plan_fft(Omega.X), plan_ifft(Omega.X)
    end

    return PrecomputedTerms(
        viscous_solver,
        loadresponse,
        fft(loadresponse),
        fourier_left_term,
        fourier_right_term,
        p1,
        p2,
    )
end


"""

    forward_isostasy!(Omega, t_out, u3D_elastic, u3D_viscous, sigma_zz, tools, p, c)

Integrates isostasy model over provided time vector `t_out` for a given
`ComputationDomain` `Omega`, pre-allocated arrays `u3D_elastic` and `u3D_viscous`
containing the solution, `sigma_zz` the vertical load applied upon the bedrock,
`tools` a set of pre-computed terms to speed up the computation, `p` some
`MultilayerEarth` parameters and `c` the `PhysicalConstants` of the problem.
"""
@inline function forward_isostasy!(
    Omega::ComputationDomain,
    t_out::AbstractVector{T},       # the output time vector
    u3D_elastic::AbstractArray{T, 3},
    u3D_viscous::AbstractArray{T, 3},
    dudt3D_viscous::AbstractArray{T, 3},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    p::MultilayerEarth,
    c::PhysicalConstants;
    dt = fill(T(years2seconds(1.0)), length(t_out)-1)::AbstractVector{T},
) where {T}

    for i in eachindex(t_out)[2:end]
        dt_out = t_out[i] - t_out[i-1]
        u3D_elastic[:, :, i] .= compute_elastic_response(Omega, tools, -sigma_zz ./ c.g )

        u3D_viscous[:, :, i], dudt3D_viscous[:, :, i] = forwardstep_isostasy(
            Omega,
            dt[i-1],
            dt_out,
            u3D_viscous[:, :, i-1],
            dudt3D_viscous[:, :, i-1],
            sigma_zz,
            tools,
            p,
        )

    end
end

"""

    forwardstep_isostasy(Omega, dt, dt_out, u_viscous, sigma_zz, tools, p)

Perform GIA computation one step forward w.r.t. the time vector used for storing the
output. This implies performing a number of in-between steps (for numerical stability
and accuracy) that are given by the ratio between `dt_out` and `dt`.
"""
@inline function forwardstep_isostasy(
    Omega::ComputationDomain,
    dt::T,
    dt_out::T,
    u_viscous::AbstractMatrix{T},
    dudt_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    p::MultilayerEarth,
) where {T<:AbstractFloat}

    if Omega.use_cuda
        u_viscous, dudt_viscous, sigma_zz = convert2CuArray([u_viscous, dudt_viscous, sigma_zz])
    end

    if tools.viscous_solver == "Euler"

        t_internal = range(0, stop = dt_out, step = dt)
        u_viscous_next = copy(u_viscous)
        dudt_viscous_next = copy(dudt_viscous)
        for i in eachindex(t_internal)[2:end]
            euler_viscous_response!(
                Omega,
                dt,
                u_viscous_next,
                dudt_viscous_next,
                sigma_zz,
                tools,
                p,
            )
            apply_bc!(u_viscous_next)
            apply_bc!(dudt_viscous_next)
        end
    
    elseif tools.viscous_solver == "CrankNicolson"
        u_viscous_next = apply_bc( cranknicolson_viscous_response(
            dt_out,
            u_viscous,
            sigma_zz,
            tools,
        ) )
    end

    if Omega.use_cuda
        u_viscous_next, dudt_viscous_next = convert2Array(
            [u_viscous_next, dudt_viscous_next])
    end


    return u_viscous_next, dudt_viscous_next
end

"""

    cranknicolson_viscous_response(dt, u_viscous, sigma_zz, tools)

Return viscous response based on Crank-Nicolson time discretization.
Only valid for solid-Earth parameters that are constant over x, y.
"""
@inline function cranknicolson_viscous_response(
    dt::T,
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
) where {T<:AbstractFloat}
    # Should: FFT the load at t = tn + Δt / 2 giving ( hat_σ_zz )_pq
    # But: If performed with small enough time step, negligible
    num = tools.fourier_right_term .* (tools.forward_fft * u_viscous) + ( tools.forward_fft * (dt .* sigma_zz) )
    return real.(tools.inverse_fft * ( num ./ tools.fourier_left_term ))
end

"""

    euler_viscous_response(Omega, dt, u_current, sigma_zz, tools, c, p)

Return viscous response based on explicit Euler time discretization and Fourier 
collocation. Valid for multilayer parameters that can vary over x, y.
"""
@inline function euler_viscous_response!(
    Omega::ComputationDomain,
    dt::T,
    u::AbstractMatrix{T},
    dudt::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    p::MultilayerEarth,
) where {T<:AbstractFloat}

    eps = 1e-20

    biharmonic_u = Omega.harmonic_coeffs .* ( tools.forward_fft * 
    ( p.lithosphere_rigidity .* (tools.inverse_fft *
    ( Omega.harmonic_coeffs .* (tools.forward_fft * u) ) ) ) )

    lgr_term = ( tools.forward_fft * sigma_zz -
    tools.forward_fft * (p.mean_density .* p.mean_gravity .* u) -
    biharmonic_u ) ./ ( Omega.pseudodiff_coeffs .+ T(eps) )

    dudt .= real.(tools.inverse_fft * lgr_term) ./ (2 .* p.effective_viscosity)
    u .+= dt .* dudt
end

"""

    apply_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at corners of domain is 0.
Whereas Bueler et al. (2007) take the edges for this computation, we take the corners
because they represent the far-field better.
"""
@inline function apply_bc(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    u_bc = copy(u)
    return apply_bc!(u_bc)
end

@inline function apply_bc!(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    CUDA.allowscalar() do
        u .-= (u[1,1] + u[1,end] + u[end,1] + u[end,end]) / T(4)
    end
end

"""

    get_cranknicolson_factors(Omega, dt, p, c)

Return two terms arising in the Crank-Nicolson scheme when applied to the present case.
"""
@inline function get_cranknicolson_factors(
    Omega::ComputationDomain,
    dt::T,
    p::MultilayerEarth,
    c::PhysicalConstants,
) where {T<:AbstractFloat}
    # Note: μ = π/L already included in differential coeffs
    beta = ( p.mantle_density .* c.g .+ p.lithosphere_rigidity .* Omega.biharmonic_coeffs )
    term1 = (2 .* p.halfspace_viscosity .* p.viscosity_scaling) .* Omega.pseudodiff_coeffs
    term2 = (dt/2) .* beta
    fourier_left_term = term1 + term2
    fourier_right_term = term1 - term2
    return fourier_left_term, fourier_right_term
end

"""

    plan_twoway_fft(X)

Return forward-FFT and inverse-FFT plan to apply on array with same dimensions as `X`.
"""
@inline function plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    return plan_fft(X), plan_ifft(X)
end

"""

    compute_elastic_response(tools, load)

Compute the elastic response of the solid Earth by convoluting the `load` with the
Green's function stored in the pre-computed `tools`(elements obtained from Farell
 1972).
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

@inline function collocate_elastic_response(
    p::MultilayerEarth,
    tools::PrecomputedTerms,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    harmonic_u = 1 ./ p.lithosphere_rigidity .* ( tools.inverse_fft * 
                 ( tools.forward_fft * load ./ tools.harmonic_coeffs ) )
    
    return real.( tools.inverse_fft * ( tools.forward_fft * harmonic_u ./ tools.harmonic_coeffs ) )
end