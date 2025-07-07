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
mutable struct SolidEarthParameters{T<:AbstractFloat, M<:KernelMatrix{T}}
    effective_viscosity::M
    litho_thickness::M
    litho_rigidity::M
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
    layer_boundaries = T.([88e3, 400e3]),
    layer_viscosities = T.([1e19, 1e21]),        # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
    litho_youngmodulus = T(DEFAULT_LITHO_YOUNGMODULUS),              # (N/m^2)
    litho_poissonratio = T(DEFAULT_LITHO_POISSONRATIO),
    mantle_poissonratio = T(DEFAULT_MANTLE_POISSONRATIO),
    tau = T(DEFAULT_MANTLE_TAU),
    rho_uppermantle = T(DEFAULT_RHO_UPPERMANTLE),   # Mean density of topmost upper mantle (kg m^-3)
    rho_litho = T(DEFAULT_RHO_LITHO),               # Mean density of lithosphere (kg m^-3)
    characteristic_loadlength = mean([domain.Wx, domain.Wy]),
    reference_viscosity = T(1e21),
) where {T<:AbstractFloat, L, M}

    if tau isa Real
        tau = fill(tau, domain.nx, domain.ny)
    end
    tau = kernelpromote(tau, domain.arraykernel)

    if layer_boundaries isa Vector{<:Real}
        layer_boundaries = matrify(layer_boundaries, domain.nx, domain.ny)
    end

    if layer_viscosities isa Vector{<:Real}
        layer_viscosities = matrify(layer_viscosities, domain.nx, domain.ny)
    end

    litho_thickness = zeros(T, domain.nx, domain.ny)
    litho_thickness .= view(layer_boundaries, :, :, 1)

    litho_rigidity = get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)
    effective_viscosity = get_effective_viscosity(domain, layer_viscosities,
        layer_boundaries, mantle_poissonratio, characteristic_loadlength,
        reference_viscosity)

    litho_thickness, litho_rigidity, effective_viscosity = kernelpromote(
        [litho_thickness, litho_rigidity, effective_viscosity], domain.arraykernel)
    
    litho_shearmodulus = litho_youngmodulus / (2 * (1 + litho_poissonratio))

    return SolidEarthParameters(
        effective_viscosity,
        litho_thickness, litho_rigidity, litho_poissonratio,
        mantle_poissonratio, tau, litho_youngmodulus, litho_shearmodulus,
        rho_uppermantle, rho_litho,
    )

end

"""
    get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}

Compute rigidity `D` based on thickness `t`, Young modulus `E` and Poisson ration `nu`.
"""
function get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}
    return (E * t^3) / (12 * (1 - nu^2))
end

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
# ViscousMantle properties
######################################################################################

abstract type AbstractViscosityLumping end

struct LocalViscosityLumping end
# For now the only one implemented so we don't bother with using the struct.
# In future, implement NeighborhoodViscosityLumping

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

"""
    get_effective_viscosity(
        layer_viscosities::Vector{KernelMatrix{T}},
        layers_thickness::Vector{T},
        domain::RegionalDomain{T, M},
    ) where {T<:AbstractFloat}

Compute equivalent viscosity for multilayer model by recursively applying
the formula for a halfspace and a channel from Lingle and Clark (1975).
"""
function get_effective_viscosity(
    domain::RegionalDomain{T, L, M},
    layer_viscosities::Array{T, 3},
    layer_boundaries::Array{T, 3},
    mantle_poissonratio::T,
    characteristic_loadlength::T,
    reference_viscosity::T = 1e21;
    correct_shearmoduluschange::Bool = true,
) where {T<:AbstractFloat, L, M}

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
            effective_viscosity .*= three_layer_scaling(domain, viscosity_ratio,
                channel_thickness, characteristic_loadlength)
        end
    end
    effective_compressible_viscosity = effective_viscosity .* compressibility_scaling

    if correct_shearmoduluschange
        corrected_viscosity = seakon_calibration(effective_compressible_viscosity,
            reference_viscosity)
    else
        corrected_viscosity = effective_compressible_viscosity
    end
    return corrected_viscosity
end

function seakon_calibration(eta::Matrix{T}, eta_ref) where {T<:AbstractFloat}
    return exp.(log10.(T(eta_ref) ./ eta)) .* eta
end

"""
    three_layer_scaling(domain::RegionalDomain, kappa::T, visc_ratio::T,
        channel_thickness::T)

Return the viscosity scaling for a three-layer model and based on a the wave
number `kappa`, the `visc_ratio` and the `channel_thickness`.
Reference: Bueler et al. 2007, below equation 15.
"""
function three_layer_scaling(
    domain::RegionalDomain{T, M},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
    characteristic_loadlength::T,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # kappa is the wavenumber of the harmonic load. (see Cathles 1975, p.43)
    # for the default value, we assume this is related to the size of the domain!
    kappa = T(π) / characteristic_loadlength

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

# E = 66.0
# He_lith = 88.0
# nu = 0.28
# D_lith = (E*1e9) * (He_lith*1e3)^3 / (12.0 * (1.0-nu^2))