rho_mantle = 3.3e3                  # kg/m^3
lithosphere_rigidity = 5e24         # N*m
mantle_viscosity = 1e21             # Pa*s

function init_solidearth_params(
    T::Type;
    rho_mantle=rho_mantle,
    lithosphere_rigidity=lithosphere_rigidity,
    mantle_viscosity=mantle_viscosity,
)
    return SolidEarthParams(T(rho_mantle), T(lithosphere_rigidity), T(mantle_viscosity))
end

struct SolidEarthParams{T<:AbstractFloat}
    rho_mantle::T
    lithosphere_rigidity::T
    mantle_viscosity::T
end