#####################################################
# Physical constants
#####################################################

g = 9.81                                # Mean Earth acceleration at surface (m/s^2)
seconds_per_year = 60^2 * 24 * 365.25   # (s)
ice_density = 0.910e3                   # (kg/m^3)
r_equator = 6.371e6                     # Earth radius at equator (m)
r_pole = 6.357e6                        # Earth radius at pole (m)
G = 6.674e-11                           # Gravity constant (m^3 kg^-1 s^-2)
mE = 5.972e24                           # Earth's mass (kg)
rho_0 = 13.1e3                          # Density of Earth's core (kg m^-3)
rho_1 = 3.0e3                           # Mean density of solid-Earth surface (kg m^-3)
# Note: rho_0 and rho_1 are chosen such that g(pole) ≈ 9.81

"""
    init_physical_constants()

Return struct containing physical constants.
"""
function init_physical_constants(;T::Type=Float64, ice_density = ice_density)
    return PhysicalConstants(
        T(g),
        T(seconds_per_year),
        T(ice_density),
        T(r_equator),
        T(r_pole),
        T(G),
        T(mE),
        T(rho_0),
        T(rho_1),
    )
end
