rho_mantle = 3.3e3                  # kg/m^3
lithosphere_rigidity = 5e24         # N*m
mantle_viscosity = 1e21             # Pa*s

channel_viscosity = 1e19            # Pa*s (Ivins 2022, Fig 10 WAIS)
halfspace_viscosity = 1e21          # Pa*s (Ivins 2022, Fig 12 WAIS)
channel_begin = 88e3                # 88 km: beginning of asthenosphere (Bueler 2007).
halfspace_begin = 400e3             # 400 km: beginning of homogenous half-space (Ivins 2022, Fig 12).


function init_solidearth_params(
    T::Type;
    rho_mantle=rho_mantle,
    lithosphere_rigidity=lithosphere_rigidity,
    mantle_viscosity=mantle_viscosity,
)
    return SolidEarthParams(T(rho_mantle), T(lithosphere_rigidity), T(mantle_viscosity))
end

# function init_solidearth_params(
#     T::Type;
#     Omega::ComputationDomain,
#     rho_mantle = rho_mantle,
#     lithosphere_rigidity = lithosphere_rigidity,
#     channel_viscosity = channel_viscosity,
#     halfspace_viscosity = halfspace_viscosity,
#     channel_begin = channel_begin,
#     halfspace_begin = halfspace_begin,
# )

#     channel_thickness = halfspace_begin - channel_begin
#     viscosity_ratio = get_viscosity_ratio(channel_viscosity, halfspace_viscosity)
#     viscosity_scaling = three_layer_scaling.(Omega.pseudodiff_coeffs, viscosity_ratio, channel_thickness)

#     return ExtendedSolidEarthParams(
#         T(rho_mantle),
#         T(lithosphere_rigidity),
#         T(channel_viscosity),
#         T(halfspace_viscosity),
#         T(viscosity_ratio),
#         T(viscosity_scaling),
#         T(channel_begin),
#         T(halfspace_begin),
#         T(channel_thickness),
#     )
# end

struct SolidEarthParams{T<:AbstractFloat}
    rho_mantle::T
    lithosphere_rigidity::T
    mantle_viscosity::T
end

# For now, assume lithosphere has always the same thickness
# struct ExtendedSolidEarthParams{T<:AbstractFloat}
#     rho_mantle::T
#     lithosphere_rigidity::T
#     channel_viscosity::Matrix{T}
#     halfspace_viscosity::Matrix{T}
#     viscosity_ratio::Matrix{T}
#     viscosity_scaling::Matrix{T}
#     channel_begin::T
#     halfspace_begin::T
#     channel_thickness::T
# end

"""

(Bueler 2007) below equation 15.
"""
function three_layer_scaling(
    kappa::T,
    visc_ratio::T,
    Tc::T,
) where {T<:AbstractFloat}

    C, S = hyperbolic_channel_coeffs(Tc, kappa)
    
    num1 = 2 * visc_ratio * C * S
    num2 = (1 - visc_ratio ^ 2) * Tc^2 * kappa ^ 2
    num3 = visc_ratio ^ 2 * S ^ 2 + C ^ 2

    denum1 = (visc_ratio + 1/visc_ratio) * C * S
    denum2 = (visc_ratio - 1/visc_ratio) * Tc * kappa
    denum3 = S^2 + C^2
    return (num1 + num2 + num3) / (denum1 + denum2 + denum3)
end

"""

(Bueler 2007) paragraph below equation 15.
"""
function get_viscosity_ratio(
    channel_viscosity::Matrix{T},
    halfspace_viscosity::Matrix{T},
) where {T<:AbstractFloat}
    return channel_viscosity ./ halfspace_viscosity
end

function hyperbolic_channel_coeffs(
    Tc::T,
    kappa::T,
) where {T<:AbstractFloat}
    return cosh(Tc * kappa), sinh(Tc * kappa)
end