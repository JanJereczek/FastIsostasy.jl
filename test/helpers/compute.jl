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
) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    mask = mask_disc(Omega.X, Omega.Y, R)
    return - mask .* (c.rho_ice * c.g * H)
end

################################################
# Generate binary parameter fields for test 3
################################################

function generate_gaussian_field(
    Omega::ComputationDomain{T, M},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::KernelMatrix{T},
) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    if Omega.Nx == Omega.Ny
        N = Omega.Nx
    else
        error("Automated generation of Gaussian parameter fields only supported for" *
            "square domains.")
    end
    G = gauss_distr( Omega.X, Omega.Y, xy_peak, sigma )
    G = G ./ maximum(G) .* z_peak
    return fill(z_background, N, N) + G
end

function slice_along_x(Omega::ComputationDomain)
    Nx, Ny = Omega.Nx, Omega.Ny
    return Nx÷2:Nx, Ny÷2
end