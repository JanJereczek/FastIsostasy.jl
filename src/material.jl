######################################################################################
# Effect of mantle compressibility on effective viscosity
######################################################################################

abstract type AbstractCompressibility end

struct CompressibleMantle end

struct IncompressibleMantle end

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

abstract type AbstractCalibration end

struct NoCalibration end

@kwdef struct SeakonCalibration{T}
    ref_viscosity::T = 1f21
end


"""
$(TYPEDSIGNATURES)

Apply the `calibration` to the viscosity `eta`.
"""
apply_calibration!(eta, calibraton::NoCalibration) = eta

function apply_calibration!(eta, calibration::SeakonCalibration)
    eta .*= exp.(log10.(T(calibration.eta_ref) ./ eta))
    return nothing
end

######################################################################################
# Lumping of 3D viscosity into effective 2D viscosity
######################################################################################

abstract type AbstractViscosityLumping end
@kwdef struct TimeDomainViscosityLumping
    characteristic_loadlength::Float32 = 2f6
end

struct FreqDomainViscosityLumping end

struct MeanViscosityLumping end

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
    R = fill(T(1), domain.nx, domain.ny)

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
    R = fill(T(1), domain.nx, domain.ny)
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
        end

    end

    return T.(effective_viscosity), T.(R)
end

function get_effective_viscosity_and_scaling(domain, layer_viscosities, layer_boundaries,
    maskactive, lumping::MeanViscosityLumping)
    T = eltype(domain.dx)
    R = fill(T(1), domain.nx, domain.ny)
    T_lithosphere = layer_boundaries[:, :, 1]
    T_uppermantle = layer_boundaries[:, :, end] .- T_lithosphere
    effective_viscosity = mean_viscosity(T_lithosphere, T_uppermantle,
        layer_viscosities, layer_boundaries)

    return T.(effective_viscosity), T.(R)
end

function get_effective_viscosity_and_scaling(domain, layer_viscosities, layer_boundaries,
    maskactive, lumping::MeanLogViscosityLumping)
    T = eltype(domain.dx)
    R = fill(T(1), domain.nx, domain.ny)
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

    # kappa is the wavenumber of the harmonic load. (see Cathles 1975, p.43)
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
    C = cosh.(channel_thickness .* kappa)
    S = sinh.(channel_thickness .* kappa)
    
    num = null(domain)
    denum = null(domain)

    @. num += 2 * visc_ratio * C * S
    @. num += (1 - visc_ratio ^ 2) * channel_thickness ^ 2 * kappa ^ 2
    @. num += visc_ratio ^ 2 * S ^ 2 + C ^ 2

    @. denum += (visc_ratio + 1 / visc_ratio) * C * S
    @. denum += (visc_ratio - 1 / visc_ratio) * channel_thickness * kappa
    @. denum += S ^ 2 + C ^ 2
    
    return num ./ denum
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
# General solid Earth properties
######################################################################################

const DEFAULT_RHO_LITHO = 3.2e3
const DEFAULT_LITHO_YOUNGMODULUS = 6.6e10
const DEFAULT_LITHO_POISSONRATIO = 0.28
const DEFAULT_LITHO_THICKNESS = 88e3
const DEFAULT_RHO_UPPERMANTLE = 3.4e3
const DEFAULT_MANTLE_POISSONRATIO = 0.28
const DEFAULT_MANTLE_TAU = 855.0

"""
    SolidEarthParameters(domain; layer_boundaries, layer_viscosities)

Return a struct containing all information related to the lateral variability of
solid-Earth parameters. To initialize with values other than default, run:

```julia
domain = RegionalDomain(3000e3, 7)
lb = [100e3, 300e3]
lv = [1e19, 1e21]
p = SolidEarthParameters(domain, layer_boundaries = lb, layer_viscosities = lv)
```

which initializes a lithosphere of thickness ``T_1 = 100 \\mathrm{km}``, a viscous
channel between ``T_1``and ``T_2 = 300 \\mathrm{km}``and a viscous halfspace starting
at ``T_2``. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `domain::RegionalDomain`.
"""
mutable struct SolidEarthParameters{
    T,  # <:AbstractFloat,
    M,  # <:KernelMatrix{T},
    B,  # <:KernelMatrix{Bool},
}
    effective_viscosity::M
    pseudodiff_scaling::M
    litho_thickness::M
    litho_rigidity::M
    maskactive::B
    litho_poissonratio::T
    mantle_poissonratio::T
    tau::M
    litho_youngmodulus::T
    litho_shearmodulus::T
    rho_uppermantle::T
    rho_litho::T
end

function SolidEarthParameters(
    domain::RegionalDomain{T, L, M};
    lumping = FreqDomainViscosityLumping(),
    compressibility = CompressibleMantle(),
    calibration = NoCalibration(),
    maskactive = domain.R .< Inf,
    layer_boundaries = T.([88e3, 400e3]),
    layer_viscosities = T.([1e19, 1e21]),        # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
    litho_youngmodulus = T(DEFAULT_LITHO_YOUNGMODULUS),              # (N/m^2)
    litho_poissonratio = T(DEFAULT_LITHO_POISSONRATIO),
    mantle_poissonratio = T(DEFAULT_MANTLE_POISSONRATIO),
    tau = T(DEFAULT_MANTLE_TAU),
    rho_uppermantle = T(DEFAULT_RHO_UPPERMANTLE),   # Mean density of topmost upper mantle (kg m^-3)
    rho_litho = T(DEFAULT_RHO_LITHO),               # Mean density of lithosphere (kg m^-3)
) where {T<:AbstractFloat, L, M}

    if tau isa Real
        tau = fill(tau, domain.nx, domain.ny)
    end
    tau = kernelpromote(tau, domain.arraykernel)

    if layer_boundaries isa Vector
        layer_boundaries = matrify(layer_boundaries, domain.nx, domain.ny)
    end

    if layer_viscosities isa Vector
        layer_viscosities = matrify(layer_viscosities, domain.nx, domain.ny)
    end

    litho_thickness = zeros(T, domain.nx, domain.ny)
    litho_thickness .= view(layer_boundaries, :, :, 1)

    litho_rigidity = get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)
    effective_viscosity, pseudodiff_scaling = get_effective_viscosity_and_scaling(
        domain, layer_viscosities, layer_boundaries, maskactive, lumping)

    apply_compressibility!(effective_viscosity, mantle_poissonratio, compressibility)
    apply_calibration!(effective_viscosity, calibration)

    litho_thickness, litho_rigidity, effective_viscosity, pseudodiff_scaling, maskactive =
        kernelpromote( [litho_thickness, litho_rigidity, effective_viscosity,
        pseudodiff_scaling, maskactive], domain.arraykernel)

    pseudodiff_scaling = 1 ./ (pseudodiff_scaling .* domain.pseudodiff)

    litho_shearmodulus = get_shearmodulus(litho_youngmodulus, litho_poissonratio)

    return SolidEarthParameters(
        effective_viscosity, pseudodiff_scaling,
        litho_thickness, litho_rigidity, kernelcollect(maskactive, domain),
        litho_poissonratio, mantle_poissonratio, tau,
        litho_youngmodulus, litho_shearmodulus, rho_uppermantle, rho_litho,
    )

end

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
get_shearmodulus(m::ReferenceSolidEarthModel) = get_shearmodulus(m.density, m.Vsv, m.Vsh)
get_shearmodulus(ρ, Vsv, Vsh) = ρ .* (Vsv + Vsh) ./ 2

######################################################################################
# Elastic properties
######################################################################################


"""
    build_greenintegrand(distance::Vector{T}, 
        greenintegrand_coeffs::Vector{T}) where {T<:AbstractFloat}

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
    get_loadgreen(r::T, rm::Vector{T}, greenintegrand_coeffs::Vector{T},     
        interp_greenintegrand_::Interpolations.Extrapolation) where {T<:AbstractFloat}

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
    get_elastic_green(domain, quad_support, quad_coeffs)

Integrate load response over field by using 2D quadrature with specified
support points and associated coefficients.
"""
function get_elastic_green(
    domain::RegionalDomain{T, M},
    greenintegrand_function::Function,
    quad_support::Vector{T},
    quad_coeffs::Vector{T},
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    dx, dy = domain.dx, domain.dy
    elasticgreen = fill(T(0), domain.nx, domain.ny)

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

"""
    get_kei(filt, L_w, dx, dy; T = Float64)

Calculate the Kelvin filter in 2D.

# Arguments
- `filt::Matrix`: The filter array to be filled with Kelvin function values.
- `L_w::Real`: The characteristic length scale.
- `dx::Real`: The grid spacing in the x-direction.
- `dy::Real`: The grid spacing in the y-direction.

# Returns
- `filt::Matrix{T}`: The filter array filled with Kelvin function values.
"""
function get_kei(domain::RegionalDomain{T, <:Any, <:Any}, L_w) where
    {T<:AbstractFloat}

    (;nx, ny, dx, dy) = domain
    if nx != ny
        error("The Kelvin filter is only implemented for square domains.")
    end
    n2 = (nx-1) ÷ 2

    # Load tabulated values and init filt before filling with loop
    rn_vals, kei_vals = load_viscous_kelvin_function(T)
    filt = null(domain)
    for j = -n2:n2
        for i = -n2:n2
            x = i*dx
            y = j*dy
            r = sqrt(x^2 + y^2)

            # Get actual index of array
            i1 = i + 1 + n2
            j1 = j + 1 + n2

            # Get correct kei value for this point
            filt[i1, j1] = get_kei_value(r, L_w, rn_vals, kei_vals)
        end
    end

    return filt
end

"""
    get_kei_value(r, L_w, rn_vals, kei_vals)

Calculate the Kelvin function (kei) value based on the radius from the point load `r`,
the flexural length scale `L_w`, and the arrays of normalized radii `rn_vals` and
corresponding kei values `kei_vals`.

This function first normalizes the radius `r` by the flexural length scale `L_w` to get
the current normalized radius. If this value is greater than the maximum value in `rn_vals`,
the function returns the maximum value in `kei_vals`. Otherwise, it finds the interval in
`rn_vals` that contains the current normalized radius and performs a linear interpolation
to calculate the corresponding kei value.
"""
function get_kei_value(r, L_w, rn_vals, kei_vals)
    n = length(rn_vals)

    # Get current normalized radius from point load
    rn_now = r / L_w

    if rn_now > rn_vals[n]
        kei = kei_vals[n]
    else
        k = 1
        while k < n
            if rn_now >= rn_vals[k] && rn_now < rn_vals[k+1]
                break
            end
            k += 1
        end

        # Linear interpolation to get current kei value
        kei = kei_vals[k] + (rn_now - rn_vals[k]) / (rn_vals[k+1] - rn_vals[k]) *
            (kei_vals[k+1] - kei_vals[k])
    end

    return kei
end

"""
    get_flexural_lengthscale(litho_rigidity, rho_uppermantle, g)

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
    calc_viscous_green(GV, kei2D, L_w, D_lith, dx, dy)

Calculate the viscous Green's function. Note that L_w contains information about
the density of the upper mantle.

"""
function calc_viscous_green(domain, litho_rigidity, kei, L_w)
    return -L_w^2 ./ (2*pi*litho_rigidity) .* kei .* (domain.dx*domain.dy)
end

"""
$(TYPEDSIGNATURES)

Convert the viscosity to relaxation times following Van Calcar et al. (in rev.).
"""
get_relaxation_time(eta, m, p) = 10^(log10(eta)*m - p)
get_relaxation_time_weaker(eta) = get_relaxation_time(eta, 0.35, 4.63)
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