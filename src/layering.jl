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

Struct to enforce uniform layering when passed to [`get_layer_boundaries`](@ref).
Contains:
- `n_layers`: the number of layers in the model.
- `boundaries`: the layer boundaries, which are constant across the domain.
"""
@kwdef struct UniformLayering{T} <: AbstractLayering{T}
    n_layers::Int = 2
    boundaries::Vector{T} = [88e3, 400e3]
end

"""
    ParallelLayering{T} <: AbstractLayering{T}

Struct to enforce parallel layering when passed to [`get_layer_boundaries`](@ref).
Contains:
- `n_layers`: the number of layers in the model.
- `thickness`: the thickness of each layer.
- `tol`: a tolerance value to add to the layer boundaries.
"""
@kwdef struct ParallelLayering{T} <: AbstractLayering{T}
    n_layers::Int = 5
    thickness::Vector{T} = fill(20e3, n_layers)
    tol::T = 0.0
end


"""
    EqualizedLayering{T} <: AbstractLayering{T}

Struct to enforce equalized layering when passed to [`get_layer_boundaries`](@ref).
Contains:
- `n_layers`: the number of layers in the model.
- `boundaries`: the layer boundaries.
- `tol`: a tolerance value to add to the layer boundaries.
"""
@kwdef struct EqualizedLayering{T} <: AbstractLayering{T}
    n_layers::Int = 3
    boundaries::Vector{T} = [88e3, 400e3]
    tol::T = 0.0
end

"""
    FoldedLayering{T} <: AbstractLayering{T}

Struct to enforce folded layering when passed to [`get_layer_boundaries`](@ref).
Contains:
- `n_layers`: the number of layers in the model.
- `max_depth`: the maximum depth of the layers.
- `tol`: a tolerance value to add to the layer boundaries.
"""
@kwdef struct FoldedLayering{T} <: AbstractLayering{T}
    n_layers::Int = 5
    max_depth::T = 350.0e3
    tol::T = 0.0
end

"""
    get_layer_boundaries(n_x, n_y, litho_thickness::Matrix{T}, layering::AbstractLayering{T})

Compute the layer boundaries for a given [`AbstractLayering`](@ref).
Output is typically passed to [`SolidEarth`](@ref) to create a layered Earth model.
"""
function get_layer_boundaries(domain::RegionalDomain, litho_thickness, layering)
    T = eltype(domain.R)
    return get_layer_boundaries(domain.nx, domain.ny, litho_thickness, layering, T)
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::UniformLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    for l in 1:layering.n_layers
        layer_boundaries[:, :, l] .= layering.boundaries[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::ParallelLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    view(layer_boundaries, :, :, 1) .= litho_thickness .+ layering.tol
    for l in 2:layering.n_layers
        view(layer_boundaries, :, :, l) .= layer_boundaries[:, :, l-1] .+ layering.thickness[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::EqualizedLayering, T)

    layer_boundaries = zeros(T, n_x, n_y, layering.n_layers)
    view(layer_boundaries, :, :, 1) .= litho_thickness .+ layering.tol
    for l in 2:layering.n_layers
        view(layer_boundaries, :, :, l) .= layering.boundaries[l]
    end
    return layer_boundaries
end

function get_layer_boundaries(n_x, n_y, litho_thickness, layering::FoldedLayering, T)

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
    Xout = zeros(T, n_x, n_y, n_l)
    itp = linear_interpolation((1:n_x, 1:n_y, z), X, extrapolation_bc = extrapolation_bc)

    for i in 1:n_x, j in 1:n_y
        view(Xout, i, j, :) .= itp.(i, j, lb[i, j, :])
    end

    return Xout
end