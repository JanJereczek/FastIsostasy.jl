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

function load_spada()
    prefix ="../data/test2/Spada/"
    cases = ["u_cap", "u_disc", "dudt_cap", "dudt_disc", "n_cap", "n_disc"]
    snapshots = ["0", "1", "2", "5", "10", "inf"]
    data = Dict{String, Vector{Matrix{Float64}}}()
    for case in cases
        tmp = Matrix{Float64}[]
        for snapshot in snapshots
            fname = string(prefix, case, "_", snapshot, ".csv")
            append!(tmp, [readdlm(fname, ',', Float64)])
        end
        data[case] = tmp
    end
    return data
end

function load_latychev(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]

    u = zeros(length(x), length(files))
    for i in eachindex(files)
        file = files[i]
        # println( file, typeof( readdlm(joinpath(dir, file), ',')[:, 1] ) )
        u[:, i] = readdlm(joinpath(dir, file), ',')[idx, 2]
    end
    u .-= u[:, 1]

    return x, u
end

#####################################################
# Idealised load cases
#####################################################
function generate_uniform_disc_load(
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants,
    R::Real,
    H::Real,
) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
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
    sigma::AbstractMatrix{T},
) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
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
