######################################################################################
# General material properties
######################################################################################

"""
$(TYPEDSIGNATURES)

Compute rigidity `D` based on thickness `t`, Young modulus `E` and Poisson ration `nu`.
"""
get_rigidity(t, E, nu) = (E * t^3) / (12 * (1 - nu^2))

"""
$(TYPEDSIGNATURES)

Compute shear modulus `G` based on Young modulus `E` and Poisson ratio `nu`, or based on
seismic velocities.
"""
get_shearmodulus(youngmodulus, poissonratio) = youngmodulus / (2 * (1 + poissonratio))
get_shearmodulus(ρ, Vsv, Vsh) = ρ .* (Vsv + Vsh) ./ 2
# get_shearmodulus(m::ReferenceSolidEarthModel) = get_shearmodulus(m.density, m.Vsv, m.Vsh)

######################################################################################
# Effect of mantle compressibility on effective viscosity
######################################################################################

"""
$(TYPEDSIGNATURES)
"""
abstract type AbstractCompressibility end

"""
$(TYPEDSIGNATURES)
"""
struct CompressibleMantle end

"""
$(TYPEDSIGNATURES)
"""
struct IncompressibleMantle end

"""
$(TYPEDSIGNATURES)
"""
apply_compressibility!(eta, nu, compressibility::IncompressibleMantle) = eta

function apply_compressibility!(eta, nu, compressibility::CompressibleMantle)
    incompressible_poissonratio = 0.5f0
    mantle_poissonratio = nu
    compressibility_scaling = (1 + incompressible_poissonratio) / (1 + mantle_poissonratio)
    eta .*= compressibility_scaling
    return nothing
end

######################################################################################
# Calibration to a specific 3D GIA model
######################################################################################

"""
$(TYPEDSIGNATURES)
"""
abstract type AbstractCalibration end

"""
$(TYPEDSIGNATURES)
"""
struct NoCalibration end

"""
$(TYPEDSIGNATURES)
"""
@kwdef struct SeakonCalibration{T}
    ref_viscosity::T = 1f21
end


"""
$(TYPEDSIGNATURES)

Apply the `calibration` to the viscosity `eta`.
"""
apply_calibration!(eta, calibraton::NoCalibration) = eta

function apply_calibration!(eta, calibration::SeakonCalibration)
    T = eltype(eta)
    eta .*= exp.(log10.(T(calibration.ref_viscosity) ./ eta))
    return nothing
end

######################################################################################
# Lumping of 3D viscosity into effective 2D viscosity
######################################################################################

"""
$(TYPEDSIGNATURES)
"""
abstract type AbstractViscosityLumping end

"""
$(TYPEDSIGNATURES)
"""
@kwdef struct TimeDomainViscosityLumping
    characteristic_loadlength::Float32 = 2f6
end

"""
$(TYPEDSIGNATURES)
"""
struct FreqDomainViscosityLumping end

"""
$(TYPEDSIGNATURES)
"""
struct MeanViscosityLumping end

"""
$(TYPEDSIGNATURES)
"""
struct MeanLogViscosityLumping end

"""
$(TYPEDSIGNATURES)

Compute equivalent viscosity for multilayer model by recursively applying
the formula for a halfspace and a channel from Lingle and Clark (1975).
"""
function get_effective_viscosity_and_scaling(domain, layer_viscosities, layer_boundaries,
    maskactive, lumping::TimeDomainViscosityLumping)

    characteristic_loadlength = lumping.characteristic_loadlength
    T = eltype(domain.dx)

    # Recursion has to start with half space = n-th layer:
    effective_viscosity = layer_viscosities[:, :, end]
    R = fill(1, domain)

    if size(layer_viscosities, 3) > 1
        channel_viscosity = layer_viscosities[:, :, end - 1]
        channel_thickness = layer_boundaries[:, :, end] - layer_boundaries[:, :, end - 1]
        viscosity_ratio = channel_viscosity ./ effective_viscosity
        
        @inbounds for l in axes(layer_viscosities, 3)[1:end-1]
            channel_viscosity .= layer_viscosities[:, :, end - l]
            channel_thickness .= layer_boundaries[:, :, end - l + 1] -
                layer_boundaries[:, :, end - l]
            viscosity_ratio = channel_viscosity ./ effective_viscosity
            effective_viscosity .*= channel_scaling_timedomain(domain, viscosity_ratio,
                channel_thickness, characteristic_loadlength)
        end
    end
    
    return T.(effective_viscosity), T.(R)
end

function get_effective_viscosity_and_scaling(domain, layer_viscosities, layer_boundaries,
    maskactive, lumping::FreqDomainViscosityLumping)

    T = eltype(domain.dx)
    R = fill(1, domain)
    effective_viscosity = layer_viscosities[:, :, end]

    if size(layer_viscosities, 3) > 1
        channel_thickness = similar(effective_viscosity)
        viscosity_ratio = similar(effective_viscosity)
        
        @inbounds for l in axes(layer_viscosities, 3)[1:end-1]
            channel_thickness .= layer_boundaries[:, :, end - l + 1] -
                layer_boundaries[:, :, end - l]
            viscosity_ratio .= layer_viscosities[:, :, end - l] ./ effective_viscosity
            R .*= channel_scaling_freqdomain_2D(domain, viscosity_ratio, channel_thickness, maskactive)
            @show extrema(R)
            if maximum(R) == typemax(eltype(R))
                error("The scaling factor R is too large. Try to introduce intermediate layers if you do not want to change the floating point precision.")
            end
        end

    end

    return T.(effective_viscosity), T.(R)
end

function get_effective_viscosity_and_scaling(domain, layer_viscosities, layer_boundaries,
    maskactive, lumping::MeanViscosityLumping)
    T = eltype(domain.dx)
    R = fill(1, domain)
    T_lithosphere = layer_boundaries[:, :, 1]
    T_uppermantle = layer_boundaries[:, :, end] .- T_lithosphere
    effective_viscosity = mean_viscosity(T_lithosphere, T_uppermantle,
        layer_viscosities, layer_boundaries)

    return T.(effective_viscosity), T.(R)
end

function get_effective_viscosity_and_scaling(domain, layer_viscosities, layer_boundaries,
    maskactive, lumping::MeanLogViscosityLumping)
    T = eltype(domain.dx)
    R = fill(1, domain)
    T_lithosphere = layer_boundaries[:, :, 1]
    T_uppermantle = layer_boundaries[:, :, end] .- T_lithosphere
    effective_viscosity = mean_viscosity(T_lithosphere, T_uppermantle,
        log10.(layer_viscosities), layer_boundaries)
    effective_viscosity = 10 .^ effective_viscosity
    return T.(effective_viscosity), T.(R)
end

"""
$(TYPEDSIGNATURES)
"""
function channel_scaling_timedomain(
    domain::RegionalDomain{T, M},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
    characteristic_loadlength::T,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # kappa is the wavenumber of the harmonic load. (see Cathles 1975, solidearth.43)
    # for the default value, we assume this is related to the size of the domain!
    kappa = T(π) / characteristic_loadlength
    return channel_scaling(domain, kappa, channel_thickness, visc_ratio)
end

function channel_scaling_freqdomain_0D(
    domain::RegionalDomain{T, M},
    visc_ratio::T,
    channel_thickness::T,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # kappa is here the pseudodiff operator in Fourier space (Bueler et al., 2007)
    kappa = Array(domain.pseudodiff)
    return channel_scaling(domain, kappa, channel_thickness, visc_ratio)
end

function channel_scaling(domain, kappa, channel_thickness, visc_ratio)
    Text = eltype(domain)
    Tint = Float64

    @show extrema(visc_ratio)
    @show extrema(channel_thickness)

    C = Tint.(cosh.(channel_thickness .* kappa))
    @show extrema(C)
    S = Tint.(sinh.(channel_thickness .* kappa))
    @show extrema(S)

    num = zeros(Tint, domain.nx, domain.ny)
    denum = copy(num)

    @. num += 2 * visc_ratio * C * S
    @show extrema(num)
    @. num += (1 - visc_ratio ^ 2) * channel_thickness ^ 2 * kappa ^ 2
    @show extrema(num)
    @. num += visc_ratio ^ 2 * S ^ 2 + C ^ 2
    @show extrema(num)

    @. denum += (visc_ratio + 1 / visc_ratio) * C * S
    @show extrema(denum)
    @. denum += (visc_ratio - 1 / visc_ratio) * channel_thickness * kappa
    @show extrema(denum)
    @. denum += S ^ 2 + C ^ 2
    @show extrema(denum)
    
    return Text.(num ./ denum)
end

function channel_scaling_freqdomain_2D(
    domain::RegionalDomain{T, M},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
    maskactive,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # actually only compute mean over maskactive
    R = channel_scaling_freqdomain_0D(
        domain,
        mean(visc_ratio[maskactive]),
        mean(channel_thickness),
    )
    return R
end

function mean_viscosity(T_lithosphere, T_uppermantle, eta, depth)
    eta_mean = similar(T_lithosphere)
    for I in CartesianIndices(eta_mean)
        eta_mean[I] = mean(eta[I, T_uppermantle[I] + T_lithosphere[I] .>= depth[I, :] .>= T_lithosphere[I]])
    end
    return eta_mean
end

######################################################################################
# Elastic properties
######################################################################################

"""
$(TYPEDSIGNATURES)

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function build_greenintegrand(
    distance::Vector{T},
    greenintegrand_coeffs::Vector{T},
) where {T<:AbstractFloat}

    greenintegrand_interp = linear_interpolation(distance, greenintegrand_coeffs)
    compute_greenintegrand_entry_r(r::T) = get_loadgreen(
        r, distance, greenintegrand_coeffs, greenintegrand_interp)
    greenintegrand_function(x::T, y::T) = compute_greenintegrand_entry_r( get_r(x, y) )
    return greenintegrand_function
end

"""
$(TYPEDSIGNATURES)

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function get_loadgreen(
    r::T,
    rm::Vector{T},
    greenintegrand_coeffs::Vector{T},
    interp_greenintegrand_::Interpolations.Extrapolation,
) where {T<:AbstractFloat}

    if r < 0.01
        return greenintegrand_coeffs[1] / ( rm[2] * T(1e12) )
    elseif r > rm[end]
        return T(0.0)
    else
        return interp_greenintegrand_(r) / ( r * T(1e12) )
    end
end

"""
$(TYPEDSIGNATURES)

Integrate load response over field by using 2D quadrature with specified
support points and associated coefficients.
"""
function get_elastic_green(
    domain::RegionalDomain,
    greenintegrand_function,
    quad_support,
    quad_coeffs,
)

    dx, dy = domain.dx, domain.dy
    elasticgreen = zeros(domain)

    @inbounds for i = 1:domain.nx, j = 1:domain.ny
        p = i - domain.mx - 1
        q = j - domain.my - 1
        elasticgreen[j, i] = quadrature2D(
            greenintegrand_function,
            quad_support,
            quad_coeffs,
            p*dx,
            (p+1)*dx,
            q*dy,
            (q+1)*dy,
        )
    end
    return elasticgreen
end

######################################################################################
# RelaxedMantle properties
######################################################################################

function besselkei(x)
    z = x * exp(im * pi / 4)
    return imag(besselk(0, z))
end

function green_viscous(domain, rho, D)
    L = get_flexural_lengthscale(D, rho, 9.81)
    R = max.(domain.R, 1)
    return map(r -> -(L^2 / (2 * pi * D) * besselkei(r / L)), R) .*
        (domain.dx * domain.dy)
end

"""
$(TYPEDSIGNATURES)

Compute the flexural length scale, based on Coulon et al. (2021), Eq. in text after Eq. 3.
The flexural length scale will be on the order of 100km.

# Arguments
- `litho_rigidity`: Lithospheric rigidity
- `rho_uppermantle`: Density of the upper mantle
- `g`: Gravitational acceleration

# Returns
- `L_w`: The calculated flexural length scale
"""
function get_flexural_lengthscale(litho_rigidity, rho_uppermantle, g)
    L_w = (litho_rigidity / (rho_uppermantle*g)) .^ 0.25
    return L_w
end

"""
$(TYPEDSIGNATURES)

Convert the viscosity to relaxation times following Van Calcar et al. (in rev.).
"""
get_relaxation_time(eta, m, p) = 10^(log10(eta)*m - p)

"""
$(TYPEDSIGNATURES)

Compute the relaxation time for a weaker mantle, following Van Calcar et al. (in rev.).
"""
get_relaxation_time_weaker(eta) = get_relaxation_time(eta, 0.35, 4.63)

"""
$(TYPEDSIGNATURES)

Compute the relaxation time for a stronger mantle, following Van Calcar et al. (in rev.).
"""
get_relaxation_time_stronger(eta) = get_relaxation_time(eta, 0.20, 1.41)

# eta1 = 1e21
# τ1_low = get_relaxation_time(eta1, 0.35, 4.63)
# τ1_high = get_relaxation_time(eta1, 0.20, 1.41)
# Gives lb, ub = 524, 616 years for 1e21, which could be a caveat compared to Spada et al. (2011)

function maxwelltime_scaling(layer_viscosities, layer_shearmoduli)
    return layer_shearmoduli[end] ./ layer_shearmoduli .* layer_viscosities
end

function maxwelltime_scaling!(layer_viscosities, layer_boundaries, m::ReferenceSolidEarthModel)
    mu = get_shearmodulus(m)
    layer_meandepths = (layer_boundaries[:, :, 1:end-1] + layer_boundaries[:, :, 2:end]) ./ 2
    layer_meandepths = cat(layer_meandepths, layer_boundaries[:, :, end], dims = 3)
    mu_itp = linear_interpolation(m.depth, mu)
    layer_meanshearmoduli = layer_viscosities ./ 1e21 .* mu_itp.(layer_meandepths)
    layer_viscosities .*= layer_meanshearmoduli[:, :, end] ./ layer_meanshearmoduli
end