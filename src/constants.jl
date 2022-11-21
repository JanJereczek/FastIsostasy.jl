g = 9.81                                    # m/s^2
seconds_per_year = 60 * 60 * 24 * 365.25    # s
rho_ice = 0.910e3                           # kg/m^3

function init_physical_constants()
    return PhysicalConstants(g, seconds_per_year, rho_ice)
end

struct PhysicalConstants{T<:AbstractFloat}
    g::T
    seconds_per_year::T
    rho_ice::T
end