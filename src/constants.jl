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
$(TYPEDFIELDS)
"""
@kwdef struct PhysicalConstants{T<:AbstractFloat}
    "Earth's mass (kg)"
    mE::T = 5.972e24
    "Earth radius at equator (m)"
    r_equator::T = 6371e3
    "Earth radius at pole (m)"
    r_pole::T = 6357e3
    "ocean surface (m²) as in Goelzer (2020), before Eq. (9)"
    A_ocean_pd::T = 3.625e14
    "mean Earth acceleration at surface (m s⁻²)"
    g::T = 9.81
    "gravity constant (m³ kg⁻¹ s⁻²)"
    G::T = 6.674e-11
    "seconds in a Julian year (s)"
    seconds_per_year::T = SECONDS_PER_YEAR
    "ice density (kg m⁻³)"
    rho_ice::T = 0.910e3
    "freshwater density (kg m⁻³)"
    rho_water::T = 1e3
    "seawater density (kg m⁻³)"
    rho_seawater::T = 1.023e3
    "`rho_seawater / rho_ice` (dimensionless)"
    rho_sw_ice::T = rho_seawater / rho_ice
end

#########################################################
# Earth model
#########################################################
"""
$(TYPEDSIGNATURES)

Define a 1D reference model of the solid Earth.

# Fields
$(TYPEDFIELDS)
"""
struct ReferenceSolidEarthModel{T<:AbstractFloat}
    "distance from Earth center"
    radius::Vector{T}
    "distance from Earth surface"
    depth::Vector{T}
    "density"
    density::Vector{T}
    "vertically polarised P-wave velocities"
    Vpv::Vector{T}
    "horizontally polarised P-wave velocities"
    Vph::Vector{T}
    "vertically polarised S-wave velocities"
    Vsv::Vector{T}
    "horizontally polarised S-wave velocities"
    Vsh::Vector{T}
end