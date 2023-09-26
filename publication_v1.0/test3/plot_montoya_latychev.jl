using CairoMakie, DelimitedFiles, NCDatasets

function load_montoya()
    file = "/home/jan/.julia/dev/FastIsostasy/data/Montoya/bedtest.nc"
    ds = NCDataset(file)
    x, y, t = ds["xc"][:, :], ds["yc"][:, :], ds["time"][:, :]
    u = ds["z_bed"][:, :]
    close(ds)
    return x, y, t, u
end

function indices_latychev2023_indices(dir::String, x_lb::Real, x_ub::Real)
    files = readdir(dir)
    x_full = readdlm(joinpath(dir, files[1]), ',')[:, 1]
    idx = x_lb .< x_full .< x_ub
    x = x_full[idx]
    return idx, x
end

function load_latychev2023(dir::String, idx)
    files = readdir(dir)
    nr = size( readdlm(joinpath(dir, files[1]), ','), 1 )
    u = zeros(nr, length(idx))
    for i in eachindex(files)
        u[:, i] = readdlm(joinpath(dir, files[i]), ',')[:, 2]
    end
    return u[idx, :]
end

function mainplot()
    seakon_file = "E0L1V1"
    lw = 5

    idx, r = indices_latychev2023_indices("../data/Latychev/$seakon_file", -1, 3e3)
    u_3DGIA = load_latychev2023("../data/Latychev/$seakon_file", idx)
    println(r)
    tplot = vcat(0:1:5, 10:5:30)

    x, y, t, u = load_montoya()
    nx, ny = length(x), length(y)
    slicex, slicey = nx÷2:nx, ny÷2
    xx = x[slicex]

    cmap = cgrad(:jet, length(tplot), categorical = true)

    fig = Figure(resolution = (3200, 2000), fontsize = 50)
    ax = Axis(fig[1, 1])
    for k in eachindex(tplot)
        lines!(ax, r, u_3DGIA[:, k], color = cmap[k], linestyle = :dash, linewidth = lw)
        lines!(ax, xx, u[slicex, slicey, k], color = cmap[k], linewidth = lw)
    end

    figfile = "plots/test3/montoya_latychev"
    save("$figfile.png", fig)
    save("$figfile.pdf", fig)
end

mainplot()