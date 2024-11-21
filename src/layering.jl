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
    boundaries::Vector{T} = [88.0, 400.0]
end

"""
    ParallelLayering{T} <: AbstractLayering{T}
"""
@kwdef struct ParallelLayering{T} <: AbstractLayering{T}
    n_layers::Int = 5
    layer_thickness::Vector{T} = fill(20.0, n_layers)
    tol::T = 0.0
end


"""
    EqualizedLayering{T} <: AbstractLayering{T}
"""
@kwdef struct EqualizedLayering{T} <: AbstractLayering{T}
    n_layers::Int = 3
    layer_thickness::Vector{T} = fill(20.0, n_layers)
    tol::T = 0.0
end

"""
    FoldedLayering{T} <: AbstractLayering{T}
"""
@kwdef struct FoldedLayering{T} <: AbstractLayering{T}
    n_layers::Int = 5
    max_depth::T = 350.0
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
    layer_boundaries[:, :, 1] .= litho_thickness .+ layering.tol
    for l in 2:layering.n_layers
        layer_boundaries[:, :, l] .= layer_boundaries[:, :, l] .+ layering.thickness[l]
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
    layer_boundaries[:, :, 1] .= litho_thickness .+ layering.tol
    for l in 2:layering.n_layers
        layer_boundaries[:, :, l] .= layer_boundaries[:, :, l-1] .+ layering.thickness[l]
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
        view(layer_boudaries, I, :) .= range(litho_thickness[I] + layering.tol,
            stop=layering.max_depth, length=layering.n_layers)
    end

    return layer_boundaries
end