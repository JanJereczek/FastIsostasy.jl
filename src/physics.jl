struct IntegratorTools{T<:AbstractFloat}
    loadresponse::AbstractMatrix{T}
    fourier_loadresponse::AbstractMatrix{Complex{T}}
    num_factor::AbstractMatrix{T}
    denum::AbstractMatrix{T}
    forward_fft::FFTW.FFTWPlan
    inverse_fft::AbstractFFTs.ScaledPlan
end

"""

    init_integrator_tools(
        dt::T,
        X::AbstractMatrix{T},
        Omega::ComputationDomain,
        p::SolidEarthParams,
    ) where {T<:AbstractFloat}

Return a `struct` containing integrator tools to perform forward-stepping. Takes a 2D
field `X`, domain parameters `Omega` and solid-Earth parameters `p` as input.
"""
@inline function init_integrator_tools(
    dt::T,
    Omega::ComputationDomain,
    p::SolidEarthParams,
    c::PhysicalConstants;
    quad_precision::Int = 4,
) where {T<:AbstractFloat}

    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    loadresponse = get_integrated_loadresponse(Omega, quad_support, quad_coeffs)
    pseudodiff_coeffs, biharmonic_coeffs = get_fourier_coeffs(T, Omega)
    num_factor, denum = get_cranknicholson_factors(
        Omega,
        dt,
        pseudodiff_coeffs,
        biharmonic_coeffs,
        p,
        c,
    )
    p1, p2 = plan_twoway_fft(Omega.X)
    return IntegratorTools(loadresponse, p1 * loadresponse, num_factor, denum, p1, p2)
end

"""

    function forwardstep_isostasy(
        dt::T,
        U::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::IntegratorTools,
    ) where {T<:AbstractFloat}

Perform forward-stepping of isostasy model given:
- `dt` the time step in seconds
- `U` the current vertical displacement
- `sigma_zz` the current vertical load
- `tools` some pre-computed tools
- `c` the physical constants of the problem.
Formula (11) of Bueler et al. 2007.
"""
@inline function forwardstep_isostasy(
    Omega::ComputationDomain,
    dt::T,
    u_elastic::AbstractMatrix{T},
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::IntegratorTools,
    c::PhysicalConstants,
) where {T<:AbstractFloat}

    # load Ψ is defined as mass per surface area --> Ψ = - σ_zz / g
    # because σ_zz = rho_ice * g * H
    u_elastic_next = compute_elastic_response(Omega, tools, -sigma_zz ./ c.g )

    # TODO FFT the load at t = tn + Δt / 2 giving ( hat_σ_zz )_pq
    num = tools.num_factor .* (tools.forward_fft * u_viscous) + ( tools.forward_fft * (dt .*sigma_zz) )
    u_viscous_next = real.(tools.inverse_fft * ( num ./ tools.denum ))   # FIXME should be symmetric
    return u_elastic_next, apply_bc(u_viscous_next)
end
# TODO make non-allocating version of this and combine with iterator.

function apply_bc(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    return u .- T( ( sum(u[1,:]) + sum(u[:,1]) ) / sum(size(u)) )
end

"""

    function forward_isostasy!(
        dt::T,
        U::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::IntegratorTools,
    ) where {T<:AbstractFloat}

Integrates isostasy model given:
- `t_vec` the time vector in seconds
- `U` the current vertical displacement
- `sigma_zz` the current vertical load
- `tools` some pre-computed tools
- `c` the physical constants of the problem.
"""
@inline function forward_isostasy!(
    Omega::ComputationDomain,
    t_vec::AbstractVector{T},
    u3D_elastic::Array{T, 3},
    u3D_viscous::Array{T, 3},
    sigma_zz::AbstractMatrix{T},
    tools::IntegratorTools,
    c::PhysicalConstants,
) where {T}

    for i in eachindex(t_vec)[2:end]
        u3D_elastic[:, :, i], u3D_viscous[:, :, i] = forwardstep_isostasy(
            Omega,
            t_vec[i]-t_vec[i-1],
            u3D_elastic[:, :, i-1],
            u3D_viscous[:, :, i-1],
            sigma_zz,
            tools,
            c,
        )
    end
end

"""

    get_fourier_coeffs(
        T::Type,
        Omega::ComputationDomain,
    )

Return coefficients resulting from transforming PDE into Fourier space.
"""
@inline function get_fourier_coeffs(
    T::Type,
    Omega::ComputationDomain,
)
    mu = T(π / Omega.L)
    raw_coeffs = mu .* T.( vcat(0:Omega.N2, Omega.N2-1:-1:0) )
    # raw_coeffs = T.( fftfreq( Omega.N, 2*Omega.L/Omega.N) )
    x_coeffs, y_coeffs = raw_coeffs, raw_coeffs
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    laplacian_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(laplacian_coeffs)
    biharmonic_coeffs = laplacian_coeffs .^ 2
    return pseudodiff_coeffs, biharmonic_coeffs
end

# TODO bloody tweak that they do!

function get_freq_coeffs(
    N::Int,
    L::T,
) where {T<:AbstractFloat}
    return fftfreq( N, L/(N*T(π)) )
end

"""

    get_cranknicholson_factors(
        Omega::ComputationDomain,
        dt::T,
        pseudodiff_coeffs,
        biharmonic_coeffs,
        p::SolidEarthParams,
    )

Return two terms arising in the Crank-Nicholson scheme when applied to thepresent case.
"""
function get_cranknicholson_factors(
    Omega::ComputationDomain,
    dt::T,
    pseudodiff_coeffs::AbstractMatrix{T},
    biharmonic_coeffs::AbstractMatrix{T},
    p::SolidEarthParams,
    c::PhysicalConstants,
) where {T<:AbstractFloat}
    # mu already included in differential coeffs
    beta = ( p.rho_mantle * c.g .+ p.lithosphere_rigidity .* biharmonic_coeffs )
    term1 = (2 * p.mantle_viscosity) .* pseudodiff_coeffs
    term2 = (dt/2) .* beta
    num_factor = term1 - term2
    denum = term1 + term2
    return num_factor, denum
end

"""

    plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}

Return forward-FFT and inverse-FFT plan to apply on array with same dimensions as `X`.
"""
function plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    return plan_fft(X), plan_ifft(X)
end

"""

    compute_elastic_response(tools::IntegratorTools, load::AbstractMatrix{T})

Compute the elastic response of the solid Earth by convoluting the load with the
Green's function (elements obtained from Farell 1972). In the Fourier space, this
corresponds to a product which is subsequently transformed back into the time domain.
Use pre-computed integration tools to accelerate computation.
"""
@inline function compute_elastic_response(
    Omega::ComputationDomain,
    tools::IntegratorTools,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return conv(load, tools.loadresponse)[Omega.N2+1:end-Omega.N2, Omega.N2+1:end-Omega.N2]
end

@inline function fft_compute_elastic_response(
    tools::IntegratorTools,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    # Note: here a element-wise multiplication is applied!
    fourier_u_elastic = tools.fourier_loadresponse .* ( tools.forward_fft * load )
    return real.( tools.inverse_fft * ( fourier_u_elastic ) )
end