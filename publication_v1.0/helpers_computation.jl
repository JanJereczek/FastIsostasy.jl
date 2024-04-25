function unpack(results::FastIsoProblem)
    Omega, p = reinit_structs_cpu(results.Omega, results.p)
    return Omega, results.c, p, results.t_out
end

function vec_dHice(Omega, Lon, Lat, t, Hitp)
    Hice_vec = [copy(Array(Omega.null)) for _ in t]
    for k in eachindex(t)
        for IJ in CartesianIndices(Lat)
            i, j = Tuple(IJ)
            # println(Omega.Lon[i, j], "  ", Omega.Lat[i, j], "  ", t[k])
            if (150 < Lon[i, j] < 180)  && (-69 < Lat[i, j] < -60)
                Hice_vec[k][i, j] = 0.0
            else
                Hice_vec[k][i, j] = Hitp(Lon[i, j], Lat[i, j], t[k])
            end
        end
    end
    deltaH = [Hice_vec[k] - Hice_vec[1] for k in eachindex(t)]
    return Hice_vec, deltaH
end

function indices_latychev2023_indices(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]
    return idx, x
end

function load_latychev_gaussian(dir::String, idx)
    files = readdir(dir)
    # nr = size( readdlm(joinpath(dir, files[1]), ','), 1 )
    nr = length(files)
    u = zeros(sum(idx), nr)
    for i in eachindex(files)
        u[:, i] = readdlm(joinpath(dir, files[i]), ',')[idx, 2]
    end
    return u
end






haf(H, b) = H + min.(b, 0) .* (1028 / 910)

function get_cellsurface(lat::Vector{T}, lon::Vector{T}) where {T<:AbstractFloat}
    R = 6371e3                     # Earth radius at equator (m)
    k = 1 ./ cos.( deg2rad.(lat) )
    dphi = mean([mean(diff(lat)), mean(diff(lon))])
    meridionallength_cell = deg2rad(dphi) * R
    azimutallength_cell = meridionallength_cell ./ k
    cellsurface = fill(meridionallength_cell, length(lon)) * azimutallength_cell'
    return cellsurface
end
