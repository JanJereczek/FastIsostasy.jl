"""
$(TYPEDSIGNATURES)

Abstract type for layering models. Subtypes should implement the `layer_boundaries` function.
Available subtypes are:
- [`UniformLayering`](@ref)
- [`ParallelLayering`](@ref)
- [`EqualizedLayering`](@ref)
- [`FoldedLayering`](@ref)
"""
abstract type AbstractLayering{T<:AbstractFloat} end

"""
$(TYPEDSIGNATURES)

Struct to enforce uniform layering when passed to [`get_layer_boundaries`](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct UniformLayering{T} <: AbstractLayering{T}
    "the number of layers in the model"
    n_layers::Int = 2
    "the layer boundaries (m), constant across the domain"
    boundaries::Vector{T} = [88e3, 400e3]
end

"""
$(TYPEDSIGNATURES)

Struct to enforce parallel layering when passed to [`get_layer_boundaries`](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ParallelLayering{T} <: AbstractLayering{T}
    "the number of layers in the model"
    n_layers::Int = 5
    "the thickness of each layer (m)"
    thickness::Vector{T} = fill(20e3, n_layers)
    "a tolerance added to the layer boundaries (m)"
    tol::T = 0.0
end


"""
$(TYPEDSIGNATURES)

Struct to enforce equalized layering when passed to [`get_layer_boundaries`](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct EqualizedLayering{T} <: AbstractLayering{T}
    "the number of layers in the model"
    n_layers::Int = 3
    "the layer boundaries (m)"
    boundaries::Vector{T} = [88e3, 400e3]
    "a tolerance added to the layer boundaries (m)"
    tol::T = 0.0
end

"""
$(TYPEDSIGNATURES)

Struct to enforce folded layering when passed to [`get_layer_boundaries`](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct FoldedLayering{T} <: AbstractLayering{T}
    "the number of layers in the model"
    n_layers::Int = 5
    "the maximum depth of the layers (m)"
    max_depth::T = 350.0e3
    "a tolerance added to the layer boundaries (m)"
    tol::T = 0.0
end

"""
$(TYPEDSIGNATURES)

Compute the layer boundaries for a given [`AbstractLayering`](@ref).
Output is typically passed to [`SolidEarth`](@ref) to create a layered Earth model.
"""
function get_layer_boundaries(domain::RegionalDomain, litho_thickness, layering)
    T = eltype(domain.R)
    return get_layer_boundaries(domain.nx, domain.ny, litho_thickness, layering, T)
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::UniformLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    for l = 1:layering.n_layers
        layer_boundaries[:, :, l] .= layering.boundaries[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::ParallelLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    view(layer_boundaries, :, :, 1) .= litho_thickness .+ layering.tol
    for l = 2:layering.n_layers
        view(layer_boundaries, :, :, l) .=
            layer_boundaries[:, :, l-1] .+ layering.thickness[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::EqualizedLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    view(layer_boundaries, :, :, 1) .= litho_thickness .+ layering.tol
    for l = 2:layering.n_layers
        view(layer_boundaries, :, :, l) .= layering.boundaries[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::FoldedLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    for I in CartesianIndices(litho_thickness)
        view(layer_boundaries, I, :) .= range(
            litho_thickness[I] + layering.tol,
            stop = layering.max_depth,
            length = layering.n_layers,
        )
    end
    return layer_boundaries
end

"""
$(TYPEDSIGNATURES)

Interpolate the values of `X` at the layer boundaries `lb` using linear interpolation.
This is typically used to interpolate the viscosity values at the layer boundaries, which
can be done by running:

```julia
z, eta3D = load_viscosities()
layer_boundaries = get_layer_boundaries(n_x, n_y, litho_thickness, layering)
layer_viscosities = 10 .^ interpolate2layers(z, log10.(eta3D), layer_boundaries)
```
"""
function interpolate2layers(
    z::Vector{T},
    X::Array{T,3},
    lb::Array{T,3};
    extrapolation_bc = Throw(),
    n_itp::Int = 4,
) where {T<:AbstractFloat}

    n_x, n_y, n_l = size(lb)
    Xout = zeros(T, n_x, n_y, n_l)
    itp =
        linear_interpolation((1:n_x, 1:n_y, z), X, extrapolation_bc = extrapolation_bc)

    for i = 1:n_x, j = 1:n_y
        view(Xout, i, j, :) .= itp.(i, j, lb[i, j, :])
    end

    return Xout
end