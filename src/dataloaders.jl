#############################################################
# Model outputs
#############################################################

function load_spada2011()
    prefix ="../testdata/Spada/"
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

function load_latychev2023(dir::String, x_lb::Real, x_ub::Real)
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
    # u .-= u[:, 1]

    return x, u
end

#############################################################
# Parameter fields
#############################################################

function load_wiens2021(Omega::ComputationDomain{T, M}; halfspace_logvisc::Real = 21) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}

    jld2file = "Wiens2021_Nx=$(Omega.Nx)_Ny=$(Omega.Ny).jld2"
    dir = "../data"
    if occursin(jld2file, readdir(dir))
        println("Preprocessed JLD2 file already exists. Skipping pre-processing.")
    else
        jld2_wiens2021(Omega.X, jld2file, dir)
    end
    @load "$jld2file" logviscosity3D logvisc_interpolators

    lv = 10.0 .^ cat( [itp.(Omega.X, Omega.Y) for itp in logvisc_interpolators]...,
        fill(T(halfspace_logvisc), Omega.Nx, Omega.Ny), dims=3)
    return lv
end

function jld2_wiens2021(Omega::ComputationDomain, jld2file::String, dir::String)

    X, Y, Nx, Ny = Omega.X, Omega.Y, Omega.Nx, Omega.Ny
    x, y = X[1,:], Y[:,1]
    rawdata = [readdlm(file) for file in readdir(dir, join = true)]
    logvisc = [filter_nan_viscosity(M) for M in rawdata]
    km2m!(logvisc)

    z = [100e3, 200e3, 300e3]
    logvisc3D = zeros(Float64, (Nx, Ny, length(z)))
    for k in axes(logvisc3D, 3)
        for i in axes(logvisc3D, 1), j in axes(logvisc3D, 2)
            logvisc3D[i, j, k] = get_closest_eta(X[i,j], Y[i,j], logvisc[k])
        end
    end

    logvisc_interpolators = [linear_interpolation( (x, y), logvisc3D[:, :, k],
        extrapolation_bc = Flat() ) for k in axes(logvisc3D, 3)]
    jldsave(jld2file, logvisc3D = logvisc3D, logvisc_interpolators = logvisc_interpolators)
    return nothing
end

function wiens_filter_nan_viscosity(M::Matrix{T}) where {T<:AbstractFloat}
    return M[.!isnan.(M[:, 3]), :]
end

function wiens_get_closest_eta(x::T, y::T, M::Matrix{T}) where {T<:AbstractFloat}
    l = argmin( (x .- M[:, 1]) .^ 2 + (y .- M[:, 2]) .^ 2 )
    return M[l, 3]
end

function km2m!(V::Vector{Matrix{T}}) where {T<:AbstractFloat}
    for i in eachindex(V)
        for j in [1, 2, 4]
            V[i][:, j] .*= T(1e3) 
        end
    end
end