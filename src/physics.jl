struct PrecomputedTerms{T<:AbstractFloat}
    loadresponse::AbstractMatrix{T}
    fourier_loadresponse::AbstractMatrix{Complex{T}}
    fourier_left_term::AbstractMatrix{T}
    fourier_right_term::AbstractMatrix{T}
    left_term::AbstractMatrix{Complex{T}}
    right_term::AbstractMatrix{Complex{T}}
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
    fourier_left_term, fourier_right_term = get_cranknicholson_factors(
        Omega,
        dt,
        p,
        c,
    )
    p1, p2 = plan_twoway_fft(Omega.X)
    left_term = p2 * fourier_left_term
    right_term = p2 * fourier_right_term
    left_realimag_ratio = get_minreal_maximag_ratio(left_term)
    right_realimag_ratio = get_minreal_maximag_ratio(left_term)
    println("Minimum real modulus divided by maximum imaginary modulus: \n
        left: $left_realimag_ratio \n
        right: $right_realimag_ratio"
    )
    return PrecomputedTerms(
        loadresponse,
        p1 * loadresponse,
        fourier_left_term,
        fourier_right_term,
        left_term,
        right_term,
        p1,
        p2,
    )
end

"""

    function forwardstep_isostasy(
        Omega::ComputationDomain,
        dt::T,
        u_viscous::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::PrecomputedTerms,
        c::PhysicalConstants,
    ) where {T<:AbstractFloat}

Forward-stepping of isostasy model based on Formula (11) of Bueler et al. 2007.
"""
@inline function forwardstep_isostasy(
    Omega::ComputationDomain,
    dt::T,
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    c::PhysicalConstants,
) where {T<:AbstractFloat}

    # load Ψ is defined as mass per surface area --> Ψ = - σ_zz / g
    # because σ_zz = rho_ice * g * H
    u_elastic_next = compute_elastic_response(Omega, tools, -sigma_zz ./ c.g ) # zeros(T, Omega.N, Omega.N) # 
    u_viscous_next = compute_viscous_response(
        dt,
        u_viscous,
        sigma_zz,
        tools,
    )

    return u_elastic_next, apply_bc(u_viscous_next)
end

function get_radial_gaussian_means(
    L::Vector{Matrix{T}},
    Omega::ComputationDomain,
    i::Int,
    j::Int,
    kernel::Function,
) where {T<:Real}
    return [radial_gaussian_mean(M, Omega.X, Omega.Y, i, j, kernel) for M in L]
end

function compute_viscous_response_heterogeneous(
    Omega::ComputationDomain,
    dt::T,
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
    p::SolidEarthParams,
) where {T<:AbstractFloat}

    u_viscous_next = zeros(size(u_viscous))
    std_dev = 100e3                         # (m)
    radial_distr(r) = radialgauss(std_dev, r)
    for i in axes(sigma_zz, 1), j in axes(sigma_zz, 2)
        
        sigma_pointload = zeros(size(sigma_zz))
        sigma_pointload[i, j] = sigma_zz[i, j]

        fields = [
            p.lithosphere_rigidity,
            p.mantle_density,
            p.halfspace_viscosity,
            p.viscosity_scaling,
        ]

        local_fields = LocalFields(get_radial_gaussian_means(
            fields,
            Omega,
            i,
            j,
            kernel,
        )...)

        fourier_left, fourier_right = get_cranknicholson_factors(
            Omega,
            dt,
            local_fields,
            c,
        )

        num = fourier_right .* (tools.forward_fft * u_viscous) +
            ( tools.forward_fft * (dt .* sigma_pointload) )
        u_viscous_next += real.(tools.inverse_fft * ( num ./ fourier_left ))
    end
    return u_viscous_next
end

struct LocalFields{T<:AbstractFloat}
    lithosphere_rigidity
    mantle_density
    halfspace_viscosity
    viscosity_scaling
end

function compute_viscous_response(
    dt::T,
    u_viscous::AbstractMatrix{T},
    sigma_zz::AbstractMatrix{T},
    tools::PrecomputedTerms,
) where {T<:AbstractFloat}
    # TODO FFT the load at t = tn + Δt / 2 giving ( hat_σ_zz )_pq
    num = tools.fourier_right_term .* (tools.forward_fft * u_viscous) + ( tools.forward_fft * (dt .* sigma_zz) )
    return real.(tools.inverse_fft * ( num ./ tools.fourier_left_term ))
end

function apply_bc(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    return u .- T( ( sum(u[1,:]) + sum(u[:,1]) ) / sum(size(u)) )
end

"""

    function forward_isostasy!(
        dt::T,
        U::AbstractMatrix{T},
        sigma_zz::AbstractMatrix{T},
        tools::PrecomputedTerms,
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
    tools::PrecomputedTerms,
    c::PhysicalConstants,
) where {T}

    for i in eachindex(t_vec)[2:end]
        u3D_elastic[:, :, i], u3D_viscous[:, :, i] = forwardstep_isostasy(
            Omega,
            t_vec[i]-t_vec[i-1],
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
    raw_coeffs = mu .* T.( vcat(0:Omega.N2, Omega.N2-1:-1:1) )
    x_coeffs, y_coeffs = raw_coeffs, raw_coeffs
    X_coeffs, Y_coeffs = meshgrid(x_coeffs, y_coeffs)
    laplacian_coeffs = X_coeffs .^ 2 + Y_coeffs .^ 2
    pseudodiff_coeffs = sqrt.(laplacian_coeffs)
    biharmonic_coeffs = laplacian_coeffs .^ 2
    return pseudodiff_coeffs, biharmonic_coeffs
end

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
    p::Union{LocalFields, SolidEarthParams},
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

function get_cranknicholson_factors(
    Omega::ComputationDomain,
    dt::T,
    pseudodiff_coeffs::AbstractMatrix{T},
    biharmonic_coeffs::AbstractMatrix{T},
    mantle_density::T,
    halfspace_viscosity::T,
    lithosphere_rigidity::T,
    c::PhysicalConstants,
) where {T<:AbstractFloat}
    # mu already included in differential coeffs
    beta = ( mantle_density .* c.g .+ lithosphere_rigidity .* biharmonic_coeffs )
    term1 = (2 .* halfspace_viscosity) .* pseudodiff_coeffs # .* p.visc_scaling
    term2 = (dt/2) .* beta
    fourier_left_term = term1 + term2
    fourier_right_term = term1 - term2
    return fourier_left_term, fourier_right_term
end

"""

    plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}

Return forward-FFT and inverse-FFT plan to apply on array with same dimensions as `X`.
"""
function plan_twoway_fft(X::AbstractMatrix{T}) where {T<:AbstractFloat}
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

@inline function fft_compute_elastic_response(
    tools::PrecomputedTerms,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    # Note: here a element-wise multiplication is applied!
    fourier_u_elastic = tools.fourier_loadresponse .* ( tools.forward_fft * load )
    return real.( tools.inverse_fft * ( fourier_u_elastic ) )
end
