struct IntegratorTools{T<:AbstractFloat}
    num_factor::Matrix{T}
    denum::Matrix{T}
    forward_fft::FFTW.FFTWPlan
    inverse_fft::AbstractFFTs.ScaledPlan
end

"""

    function forwardstep_isostasy(
        dt::T,
        U::Matrix{T},
        sigma_zz::Matrix{T},
        tools::IntegratorTools,
    ) where {T<:AbstractFloat}

Perform forward-stepping of isostasy models given:
- `dt` the time step in seconds
- `U` the current vertical displacement
- `sigma_zz` the current vertical load
- `tools` some pre-computed tools.
Formula (11) of Bueler et al. 2007.
"""
@inline function forwardstep_isostasy(
    dt::T,
    U::Matrix{T},
    sigma_zz::Matrix{T},
    tools::IntegratorTools,
) where {T<:AbstractFloat}

    num = tools.num_factor .* (tools.forward_fft * U) + dt .* (tools.forward_fft * sigma_zz )
    return real.(tools.inverse_fft * ( num ./ tools.denum ))
end

@inline function forward_isostasy!(
    t_vec::AbstractVector{T},
    u3D::Array{T, 3},
    sigma_zz::Matrix{T},
    tools::IntegratorTools,
) where {T}

    for i in eachindex(t_vec)[2:end]
        u3D[:, :, i] = forwardstep_isostasy(
            t_vec[i]-t_vec[i-1],
            u3D[:, :, i-1],
            sigma_zz,
            tools,
        )
    end
end
"""

    init_integrator_tools(
        dt::T,
        X::Matrix{T},
        d::DomainParams,
        p::SolidEarthParams,
    ) where {T<:AbstractFloat}

Return a `struct` containing integrator tools to perform forward-stepping. Takes a 2D
field `X`, domain parameters `d` and solid-Earth parameters `p` as input.
"""
function init_integrator_tools(
    dt::T,
    X::Matrix{T},
    d::DomainParams,
    p::SolidEarthParams,
    c::PhysicalConstants,
) where {T<:AbstractFloat}

    pseudodiff_coeffs, biharmonic_coeffs = get_fourier_coeffs(T, d)
    num_factor, denum = get_cranknicholson_factors(
        dt,
        pseudodiff_coeffs,
        biharmonic_coeffs,
        p,
        c,
    )
    p1, p2 = plan_twoway_fft(X)
    return IntegratorTools(num_factor, denum, p1, p2)
end

"""

    get_fourier_coeffs(
        T::Type,
        d::DomainParams,
    )

Return coefficients resulting from transforming PDE into Fourier space.
"""
function get_fourier_coeffs(
    T::Type,
    d::DomainParams,
)
    raw_coeffs = T.( (π/d.L) .* vcat(0:d.N/2, d.N/2-1:-1:1) )
    # x_coeffs, y_coeffs = get_freq_coeffs(d.N, d.L), get_freq_coeffs(d.N, d.L)
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
        dt::T,
        pseudodiff_coeffs,
        biharmonic_coeffs,
        p::SolidEarthParams,
    )

Return two terms arising in the Crank-Nicholson scheme when applied to thepresent case.
"""
function get_cranknicholson_factors(
    dt::T,
    pseudodiff_coeffs::Matrix{T},
    biharmonic_coeffs::Matrix{T},
    p::SolidEarthParams,
    c::PhysicalConstants,
) where {T<:AbstractFloat}
    term1 = (2 * p.mantle_viscosity) .* pseudodiff_coeffs
    term2 = (dt/2) .* ( p.rho_mantle * c.g .+ p.lithosphere_rigidity .* biharmonic_coeffs )
    num_factor = term1 - term2
    denum = term1 + term2
    return num_factor, denum
end

"""

    plan_twoway_fft(X::Matrix{T}) where {T<:AbstractFloat}

Return forward-FFT and inverse-FFT plan to apply on array with same dimensions as `X`.
"""
function plan_twoway_fft(X::Matrix{T}) where {T<:AbstractFloat}
    return plan_fft(X), plan_ifft(X)
end