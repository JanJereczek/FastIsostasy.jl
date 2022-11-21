rho_mantle = 3.3e3                  # kg/m^3
lithosphere_rigidity = 5e24         # N*m
mantle_viscosity = 1e21             # Pa*s

function init_solidearth_params(;
    rho_mantle::T=rho_mantle,
    lithosphere_rigidity::T=lithosphere_rigidity,
    mantle_viscosity::T=mantle_viscosity,
) where {T<:AbstractFloat}
    return SolidEarthParams(rho_mantle, lithosphere_rigidity, mantle_viscosity)
end

struct SolidEarthParams{T<:AbstractFloat}
    rho_mantle::T
    lithosphere_rigidity::T
    mantle_viscosity::T
end