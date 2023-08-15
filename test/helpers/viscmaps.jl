using DelimitedFiles
using JLD2
using CairoMakie
using Interpolations


function get2Dinterpolator(M::Matrix{T}) where {T<:AbstractFloat}
    return linear_interpolation( (M[:, 1], M[:, 2]), M[:, 3] )
end

function km2m!(V::Vector{Matrix{T}}) where {T<:AbstractFloat}
    for i in eachindex(V)
        for j in [1, 2, 4]
            V[i][:, j] .*= T(1e3) 
        end
    end
end



function get_closest_eta(x::T, y::T, M::Matrix{T}) where {T<:AbstractFloat}
    l = argmin( (x .- M[:, 1]).^2 + (y .- M[:, 2]).^2 )
    return M[l, 3]
end

function interpolate_viscosity_xy(X, Y, Eta, Eta_mean)
    x, y = X[1,:], Y[:,1]
    eta_interpolators = [linear_interpolation(
        (x, y),
        Eta[:, :, k],
        extrapolation_bc = Flat(),
    ) for k in axes(Eta, 3)]
    eta_mean_interpolator = linear_interpolation(
        (x, y),
        Eta_mean[:, :, 1],
        extrapolation_bc = Flat(),
    )
    jldsave(
        "../data/wiens_viscosity_map.jld2",
        eta = Eta,
        eta_mean = Eta_mean[:, :, 1],
        eta_interpolators = eta_interpolators,
        eta_mean_interpolator = eta_mean_interpolator,
    )
    return eta_interpolators, eta_mean_interpolator
end

function viscositytxt2matrix()
    dx = 40e3   # wiens worked on 40km resolution
    W = 3000e3  # Remap this on 2L x 2L domain
    x = collect(range( -W, stop = W, step = dx ))
    y = copy( x )
    X, Y = meshgrid(x, y)
    Eta, Eta_mean, z = load_wiens2021(X, Y)
    interpolate_viscosity_xy(X, Y, Eta, Eta_mean)

    fig = Figure(resolution = (1600, 600))
    axs = [Axis(
        fig[2, j],
        title = string(Int(z[j]/1e3), " km"),
        xlabel = "x (1000 km)",
        ylabel = (j == 1 ? "y (1000 km)" : " "),
        yticklabelsvisible = (j == 1 ? true : false),
        aspect = DataAspect(),
    ) for j in eachindex(z)]
    ax_mean = Axis(
        fig[2, 4],
        title = "Mean over {100, 200, 300} km",
        xlabel = "x (1000 km)",
        yticklabelsvisible = false,
        aspect = DataAspect(),
    )
    cmap = cgrad(:jet, rev = true)
    crange = (18, 23)
    for k in axes(Eta, 3)
        heatmap!(
            axs[k],
            X./1e6,
            Y./1e6,
            Eta[:, :, k],
            colorrange = (18, 23),
            colormap = :jet,
        )
    end
    heatmap!(
        ax_mean,
        X./1e6,
        Y./1e6,
        Eta_mean[:, :, 1],
        colorrange = (18, 23),
        colormap = :jet,
    )

    Colorbar(
        fig[1, :],
        colorrange = crange,
        colormap = cmap,
        vertical = false,
        width = Relative(0.4),
        lowclip = cmap[1],
        highclip = cmap[end],
        label = "Log10 viscosity (Pa s)",
    )
    save("plots/wiens_visc_map.png", fig)
    save("plots/wiens_visc_map.pdf", fig)

end

function interpolate_visc_wiens_on_grid(X, Y)
    eta_mean_interpolator = load(
        "../data/visc_field/doug_viscosity_map.jld2",
        "eta_mean_interpolator",
    )
    return eta_mean_interpolator.(X, Y)
end
