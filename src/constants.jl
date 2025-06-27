#########################################################
# Physical constants
#########################################################
"""
    PhysicalConstants

Return a struct containing important physical constants.
Comes with default values that can however be changed by the user, for instance by running:

```julia
c = PhysicalConstants(rho_ice = 0.93)   # (kg/m^3)
```

All constants are given in SI units (kilogram, meter, second).
"""
@kwdef struct PhysicalConstants{T<:AbstractFloat}
    mE::T = 5.972e24                        # Earth's mass (kg)
    r_equator::T = 6371e3                   # Earth radius at equator (m)
    r_pole::T = 6357e3                      # Earth radius at pole (m)
    A_ocean_pd::T = 3.625e14                # Ocean surface (m^2) as in Goelzer (2020) before Eq. (9)
    g::T = 9.8                              # Mean Earth acceleration at surface (m/s^2)
    G::T = 6.674e-11                        # Gravity constant (m^3 kg^-1 s^-2)
    seconds_per_year::T = SECONDS_PER_YEAR  # (s)
    rho_ice::T = 0.910e3                    # (kg/m^3)
    rho_water::T = 1e3                      # (kg/m^3)
    rho_seawater::T = 1.023e3               # (kg/m^3)
    rho_sw_ice::T = rho_seawater / rho_ice  # (dimensionless)
end

#########################################################
# Earth model
#########################################################
"""
    ReferenceSolidEarthModel
Return a struct with vectors containing the:
 - radius (distance from Earth center),
 - depth (distance from Earth surface),
 - density,
 - P-wave velocities,
 - S-wave velocities,
which are typically used to characterize the properties of a spherically symmetrical solid Earth.
"""
struct ReferenceSolidEarthModel{T<:AbstractFloat}
    radius::Vector{T}
    depth::Vector{T}
    density::Vector{T}
    Vpv::Vector{T}
    Vph::Vector{T}
    Vsv::Vector{T}
    Vsh::Vector{T}
end