#####################################################
# Physical constants
#####################################################

mE = 5.972e24                           # Earth's mass (kg)
r_equator = 6.371e6                     # Earth radius at equator (m)
r_pole = 6.357e6                        # Earth radius at pole (m)
A_ocean = 3.625e14                      # Ocean surface (m) as in Goelzer (2020) before Eq. (9)
g = 9.81                                # Mean Earth acceleration at surface (m/s^2)
G = 6.674e-11                           # Gravity constant (m^3 kg^-1 s^-2)
seconds_per_year = 60^2 * 24 * 365.25   # (s)
rho_ice = 0.910e3                   # (kg/m^3)
rho_water = 1.023e3                 # (kg/m^3)
rho_seawater = 1.023e3              # (kg/m^3)
rho_core = 13.1e3                        # Density of Earth's core (kg m^-3)
rho_topastheno = 3.3e3                           # Mean density of solid-Earth surface (kg m^-3)
rho_litho = 3.0e3                       # Mean density of solid-Earth surface (kg m^-3)
# Note: rho_0 and rho_1 are chosen such that g(pole) ≈ 9.81

"""
    PhysicalConstants()

Return struct containing physical constants.
"""
function PhysicalConstants(;T::Type=Float64, rho_ice = rho_ice)
    return PhysicalConstants(
        T(mE),
        T(r_equator),
        T(r_pole),
        T(A_ocean),
        T(g),
        T(G),
        T(seconds_per_year),
        T(rho_ice),
        T(rho_water),
        T(rho_seawater),
        T(rho_core),
        T(rho_topastheno),
    )
end
