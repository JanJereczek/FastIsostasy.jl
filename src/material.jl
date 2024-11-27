
"""
    get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}

Compute rigidity `D` based on thickness `t`, Young modulus `E` and Poisson ration `nu`.
"""
function get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}
    return (E * t^3) / (12 * (1 - nu^2))
end

get_shearmodulus(m::ReferenceEarthModel) = get_shearmodulus(m.density, m.Vsv, m.Vsh)
get_shearmodulus(ρ, Vsv, Vsh) = ρ .* (Vsv + Vsh) ./ 2

function maxwelltime_scaling(layer_viscosities, layer_shearmoduli)
    return layer_shearmoduli[end] ./ layer_shearmoduli .* layer_viscosities
end

function maxwelltime_scaling!(layer_viscosities, layer_boundaries, m::ReferenceEarthModel)
    mu = get_shearmodulus(m)
    layer_meandepths = (layer_boundaries[:, :, 1:end-1] + layer_boundaries[:, :, 2:end]) ./ 2
    layer_meandepths = cat(layer_meandepths, layer_boundaries[:, :, end], dims = 3)
    mu_itp = linear_interpolation(m.depth, mu)
    layer_meanshearmoduli = layer_viscosities ./ 1e21 .* mu_itp.(layer_meandepths)
    layer_viscosities .*= layer_meanshearmoduli[:, :, end] ./ layer_meanshearmoduli
end

"""
    get_effective_viscosity(
        layer_viscosities::Vector{KernelMatrix{T}},
        layers_thickness::Vector{T},
        Omega::ComputationDomain{T, M},
    ) where {T<:AbstractFloat}

Compute equivalent viscosity for multilayer model by recursively applying
the formula for a halfspace and a channel from Lingle and Clark (1975).
"""
function get_effective_viscosity(
    Omega::ComputationDomain{T, M},
    layer_viscosities::Array{T, 3},
    layer_boundaries::Array{T, 3},
    mantle_poissonratio::T;
    correct_shearmoduluschange::Bool = true,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    incompressible_poissonratio = T(0.5)
    compressibility_scaling = (1 + incompressible_poissonratio) / (1 + mantle_poissonratio)

    # Recursion has to start with half space = n-th layer:
    effective_viscosity = layer_viscosities[:, :, end]

    if size(layer_viscosities, 3) > 1
        channel_viscosity = layer_viscosities[:, :, end - 1]
        channel_thickness = layer_boundaries[:, :, end] - layer_boundaries[:, :, end - 1]
        viscosity_ratio = channel_viscosity ./ effective_viscosity
        
        @inbounds for l in axes(layer_viscosities, 3)[1:end-1]
            channel_viscosity .= layer_viscosities[:, :, end - l]
            channel_thickness .= layer_boundaries[:, :, end - l + 1] -
                layer_boundaries[:, :, end - l]
            viscosity_ratio = channel_viscosity ./ effective_viscosity
            effective_viscosity .*= three_layer_scaling(Omega, viscosity_ratio,
                channel_thickness)
        end
    end
    effective_compressible_viscosity = effective_viscosity .* compressibility_scaling

    if correct_shearmoduluschange
        corrected_viscosity = seakon_calibration(effective_compressible_viscosity)
    else
        corrected_viscosity = effective_compressible_viscosity
    end
    return corrected_viscosity
end

function seakon_calibration(eta::Matrix{T}) where {T<:AbstractFloat}
    return exp.(log10.(T(1e21) ./ eta)) .* eta
end

"""
    three_layer_scaling(Omega::ComputationDomain, kappa::T, visc_ratio::T,
        channel_thickness::T)

Return the viscosity scaling for a three-layer model and based on a the wave
number `kappa`, the `visc_ratio` and the `channel_thickness`.
Reference: Bueler et al. 2007, below equation 15.
"""
function three_layer_scaling(
    Omega::ComputationDomain{T, M},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # kappa is the wavenumber of the harmonic load. (see Cathles 1975, p.43)
    # we assume this is related to the size of the domain!
    kappa = T(π) / mean([Omega.Wx, Omega.Wy])

    C = cosh.(channel_thickness .* kappa)
    S = sinh.(channel_thickness .* kappa)
    
    num = null(Omega)
    denum = null(Omega)

    @. num += 2 * visc_ratio * C * S
    @. num += (1 - visc_ratio ^ 2) * channel_thickness ^ 2 * kappa ^ 2
    @. num += visc_ratio ^ 2 * S ^ 2 + C ^ 2

    @. denum += (visc_ratio + 1 / visc_ratio) * C * S
    @. denum += (visc_ratio - 1 / visc_ratio) * channel_thickness * kappa
    @. denum += S ^ 2 + C ^ 2
    
    return num ./ denum
end