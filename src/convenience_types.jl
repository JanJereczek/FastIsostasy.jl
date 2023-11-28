KernelMatrix{T} = Union{Matrix{T}, CuMatrix{T}} where {T<:AbstractFloat}
ComplexMatrix{T} = Union{Matrix{C}, CuMatrix{C}} where {T<:AbstractFloat, C<:Complex{T}}
BoolMatrix{T} = Union{Matrix{Bool}, CuMatrix{Bool}}

"""
    ForwardPlan

Allias for in-place precomputed plans from FFTW or CUFFT. Used to compute forward FFT.
"""
ForwardPlan{T} = Union{
    cFFTWPlan{Complex{T}, -1, true, 2, Tuple{Int64, Int64}}, 
    CUFFT.cCuFFTPlan{Complex{T}, -1, true, 2}
} where {T<:AbstractFloat}

"""
    InversePlan

Allias for in-place precomputed plans from FFTW or CUFFT. Used to compute inverse FFT.
"""
InversePlan{T} = Union{
    AbstractFFTs.ScaledPlan{Complex{T}, cFFTWPlan{Complex{T}, 1, true, 2, UnitRange{Int64}}, T},
    AbstractFFTs.ScaledPlan{Complex{T}, CUFFT.cCuFFTPlan{Complex{T}, 1, true, 2}, T}
} where {T<:AbstractFloat}