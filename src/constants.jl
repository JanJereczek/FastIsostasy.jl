#########################################################
# Physical constants
#########################################################
"""
$(TYPEDSIGNATURES)

Carry physical constants, with default values that can be changed by the user.
For instance:

```julia
c = PhysicalConstants(rho_ice = 0.93)   # (kg/m^3)
```

All constants are given in SI units (kilogram, meter, second).

# Fields
- `mE`: Earth's mass (kg)
- `r_equator`: Earth radius at equator (m)
- `r_pole`: Earth radius at pole (m)
- `A_ocean_pd`: Ocean surface (m^2) as in Goelzer (2020) before Eq. (9)
- `g`: Mean Earth acceleration at surface (m/s^2)
- `G`: Gravity constant (m^3 kg^-1 s^-2)
- `seconds_per_year`: (s)
- `rho_ice`: (kg/m^3)
- `rho_water`: (kg/m^3)
- `rho_seawater`: (kg/m^3)
- `rho_sw_ice`: (dimensionless)
"""
@kwdef struct PhysicalConstants{T<:AbstractFloat}
    mE::T = 5.972e24
    r_equator::T = 6371e3
    r_pole::T = 6357e3
    A_ocean_pd::T = 3.625e14
    g::T = 9.81
    G::T = 6.674e-11
    seconds_per_year::T = SECONDS_PER_YEAR
    rho_ice::T = 0.910e3
    rho_water::T = 1e3
    rho_seawater::T = 1.023e3
    rho_sw_ice::T = rho_seawater / rho_ice  # (dimensionless)
end

#########################################################
# Earth model
#########################################################
"""
$(TYPEDSIGNATURES)

Define a 1D reference model of the solid Earth.

# Fields
- `radius`: distance from Earth center,
- `depth`: distance from Earth surface,
- `density`: density,
- `Vpv`: P-wave velocities,
- `Vph`: P-wave velocities,
- `Vsv`: S-wave velocities,
- `Vsh`: S-wave velocities,
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