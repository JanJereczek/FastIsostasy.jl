"""
    AbstractLayering{T<:AbstractFloat}

Abstract type for layering models. Subtypes should implement the `layer_boundaries` function.
Available subtypes are:
- [`UniformLayering`](@ref)
- [`ParallelLayering`](@ref)
- [`EqualizedLayering`](@ref)
- [`FoldedLayering`](@ref)
"""
abstract type AbstractLayering{T<:AbstractFloat} end

"""
    UniformLayering{T} <: AbstractLayering{T}
"""
@kwdef struct UniformLayering{T} <: AbstractLayering{T}
    n_layers::Int = 2
    boundaries::Vector{T} = [88e3, 400e3]
end

"""
    ParallelLayering{T} <: AbstractLayering{T}
"""
@kwdef struct ParallelLayering{T} <: AbstractLayering{T}
    n_layers::Int = 5
    thickness::Vector{T} = fill(20e3, n_layers)
    tol::T = 0.0
end


"""
    EqualizedLayering{T} <: AbstractLayering{T}
"""
# @kwdef struct EqualizedLayering{T} <: AbstractLayering{T}
#     n_layers::Int = 3
#     layer_thickness::Vector{T} = fill(20e3, n_layers)
#     tol::T = 0.0
# end

@kwdef struct EqualizedLayering{T} <: AbstractLayering{T}
    n_layers::Int = 3
    boundaries::Vector{T} = [88e3, 400e3]
    tol::T = 0.0
end

"""
    FoldedLayering{T} <: AbstractLayering{T}
"""
@kwdef struct FoldedLayering{T} <: AbstractLayering{T}
    n_layers::Int = 5
    max_depth::T = 350.0e3
    tol::T = 0.0
end

"""
    get_layer_boundaries(n_x, n_y, litho_thickness::Matrix{T}, layering::AbstractLayering{T})

Compute the layer boundaries for a given layering model.
"""
function get_layer_boundaries(
    n_x::Int,
    n_y::Int,
    litho_thickness::Matrix{T},
    layering::UniformLayering{T},
) where {T<:AbstractFloat}
    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    for l in 1:layering.n_layers
        layer_boundaries[:, :, l] .= layering.boundaries[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(
    n_x::Int,
    n_y::Int,
    litho_thickness::Matrix{T},
    layering::ParallelLayering{T},
) where {T<:AbstractFloat}
    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    view(layer_boundaries, :, :, 1) .= litho_thickness .+ layering.tol
    for l in 2:layering.n_layers
        view(layer_boundaries, :, :, l) .= layer_boundaries[:, :, l-1] .+ layering.thickness[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(
    n_x::Int,
    n_y::Int,
    litho_thickness::Matrix{T},
    layering::EqualizedLayering{T},
) where {T<:AbstractFloat}
    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    view(layer_boundaries, :, :, 1) .= litho_thickness .+ layering.tol
    for l in 2:layering.n_layers
        # layer_boundaries[:, :, l] .= maximum(layer_boundaries[:, :, l-1]) .+
        #     layering.thickness[l]
        view(layer_boundaries, :, :, l) .= layering.boundaries[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(
    n_x::Int,
    n_y::Int,
    litho_thickness::Matrix{T},
    layering::FoldedLayering{T},
) where {T<:AbstractFloat}
    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    
    for I in CartesianIndices(litho_thickness)
        view(layer_boundaries, I, :) .= range(litho_thickness[I] + layering.tol,
            stop=layering.max_depth, length=layering.n_layers)
    end

    return layer_boundaries
end

"""
    interpolate2layers(z, X, lb)

Interpolate the values of `X` at the layer boundaries `lb` using linear interpolation.
This is typically used to interpolate the viscosity values at the layer boundaries, which
can be done by running:

```julia
z, eta3D = load_viscosities()
layer_boundaries = get_layer_boundaries(n_x, n_y, litho_thickness, layering)
layer_viscosities = 10 .^ interpolate2layers(z, log10.(eta3D), layer_boundaries)
```
"""
function interpolate2layers(z::Vector{T}, X::Array{T, 3}, lb::Array{T, 3};
    extrapolation_bc = Throw(), n_itp::Int = 4) where {T<:AbstractFloat}

    n_x, n_y, n_l = size(lb)
    # X_itp = zeros(n_x, n_y, n_l * n_itp)
    # lb_itp = zeros(n_x, n_y, n_l * n_itp)

    Xout = zeros(T, n_x, n_y, n_l)
    itp = linear_interpolation((1:n_x, 1:n_y, z), X, extrapolation_bc = extrapolation_bc)

    for i in 1:n_x, j in 1:n_y
        view(Xout, i, j, :) .= itp.(i, j, lb[i, j, :])
    end

    return Xout
end

const DEFAULT_RHO_LITHO = 3.2e3
const DEFAULT_LITHO_YOUNGMODULUS = 6.6e10
const DEFAULT_LITHO_POISSONRATIO = 0.28
const DEFAULT_LITHO_THICKNESS = 88e3
const DEFAULT_RHO_UPPERMANTLE = 3.4e3
const DEFAULT_MANTLE_POISSONRATIO = 0.28
const DEFAULT_MANTLE_TAU = 855.0

"""
    LayeredEarth(Omega; layer_boundaries, layer_viscosities)

Return a struct containing all information related to the lateral variability of
solid-Earth parameters. To initialize with values other than default, run:

```julia
Omega = ComputationDomain(3000e3, 7)
lb = [100e3, 300e3]
lv = [1e19, 1e21]
p = LayeredEarth(Omega, layer_boundaries = lb, layer_viscosities = lv)
```

which initializes a lithosphere of thickness ``T_1 = 100 \\mathrm{km}``, a viscous
channel between ``T_1``and ``T_2 = 200 \\mathrm{km}``and a viscous halfspace starting
at ``T_2``. This represents a homogenous case. For heterogeneous ones, simply make
`lb::Vector{Matrix}`, `lv::Vector{Matrix}` such that the vector elements represent the
lateral variability of each layer on the grid of `Omega::ComputationDomain`.
"""
mutable struct LayeredEarth{T<:AbstractFloat, M<:KernelMatrix{T}}
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

function LayeredEarth(
    Omega::ComputationDomain{T, L, M};
    layer_boundaries::A = T.([88e3, 400e3]),
    layer_viscosities::B = T.([1e19, 1e21]),        # (Pa*s) (Bueler 2007, Ivins 2022, Fig 12 WAIS)
    litho_youngmodulus::T = T(DEFAULT_LITHO_YOUNGMODULUS),              # (N/m^2)
    litho_poissonratio::T = T(DEFAULT_LITHO_POISSONRATIO),
    mantle_poissonratio::T = T(DEFAULT_MANTLE_POISSONRATIO),
    tau::T = T(DEFAULT_MANTLE_TAU),
    rho_uppermantle::T = T(DEFAULT_RHO_UPPERMANTLE),   # Mean density of topmost upper mantle (kg m^-3)
    rho_litho::T = T(DEFAULT_RHO_LITHO),               # Mean density of lithosphere (kg m^-3)
    characteristic_loadlength::T = mean([Omega.Wx, Omega.Wy]),
    reference_viscosity::T = T(1e21),
) where {
    T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T},
    A<:Union{Vector{T}, Array{T, 3}},
    B<:Union{Vector{T}, Array{T, 3}},
}

    if tau isa Real
        tau = fill(tau, Omega.nx, Omega.ny)
    end
    tau = kernelpromote(tau, Omega.arraykernel)

    if layer_boundaries isa Vector{<:Real}
        layer_boundaries = matrify(layer_boundaries, Omega.nx, Omega.ny)
    end

    if layer_viscosities isa Vector{<:Real}
        layer_viscosities = matrify(layer_viscosities, Omega.nx, Omega.ny)
    end

    litho_thickness = zeros(T, Omega.nx, Omega.ny)
    litho_thickness .= view(layer_boundaries, :, :, 1)

    litho_rigidity = get_rigidity.(litho_thickness, litho_youngmodulus, litho_poissonratio)
    effective_viscosity = get_effective_viscosity(Omega, layer_viscosities,
        layer_boundaries, mantle_poissonratio, characteristic_loadlength,
        reference_viscosity)

    litho_thickness, litho_rigidity, effective_viscosity = kernelpromote(
        [litho_thickness, litho_rigidity, effective_viscosity], Omega.arraykernel)
    
    litho_shearmodulus = litho_youngmodulus / (2 * (1 + litho_poissonratio))

    return LayeredEarth(
        effective_viscosity,
        litho_thickness, litho_rigidity, litho_poissonratio,
        mantle_poissonratio, tau, litho_youngmodulus, litho_shearmodulus,
        rho_uppermantle, rho_litho,
    )

end