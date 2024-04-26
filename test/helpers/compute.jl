function inn(X)
    return X[2:end-1, 2:end-1]
end

#####################################################
# Data loaders
#####################################################
function interpolate_spada_benchmark(c, data)
    idx = sortperm(data[:, 1])
    theta = data[:, 1][idx]
    z = data[:, 2][idx]
    x = deg2rad.(theta) .* c.r_equator
    itp = linear_interpolation(x, z, extrapolation_bc = Flat())
    return itp
end

#####################################################
# Idealised load cases
#####################################################
function generate_uniform_disc_load(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants,
    R::Real,
    H::Real,
) where {T<:AbstractFloat, M<:Matrix{T}}
    mask = mask_disc(Omega.X, Omega.Y, R)
    return - mask .* (c.rho_ice * c.g * H)
end

################################################
# Generate binary parameter fields for test 3
################################################

function slice_along_x(Omega)
    Nx, Ny = Omega.Nx, Omega.Ny
    return Nx÷2:Nx, Ny÷2
end
