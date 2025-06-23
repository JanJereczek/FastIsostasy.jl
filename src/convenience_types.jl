KernelMatrix{T} = Union{Matrix{T}, CuMatrix{T}} where {T<:AbstractFloat}
ComplexMatrix{T} = Union{Matrix{C}, CuMatrix{C}} where {T<:AbstractFloat, C<:Complex{T}}
BoolMatrix = Union{Matrix{Bool}, CuMatrix{Bool}}

ODEsolvers = Union{Tsit5, Euler, SplitEuler, Heun, Ralston,
    Midpoint, RK4, BS3, OwrenZen3, OwrenZen4, OwrenZen5, BS5, DP5, Anas5,
    RKO65, FRK65, RKM, MSRK5, MSRK6, PSRK4p7q6, PSRK3p5q4, PSRK3p6q5,
    Stepanov5, SIR54, Alshina2, Alshina3, Alshina6}

"""
    ForwardPlan

Allias for in-place precomputed plans from FFTW or CUFFT. Used to compute forward FFT.
"""
ForwardPlan{T} = Union{
    cFFTWPlan{Complex{T}, -1, true, 2, Tuple{Int64, Int64}},
    CUFFT.CuFFTPlan{Complex{T}, Complex{T}, -1, true, 2}
} where {T<:AbstractFloat}

"""
    InversePlan

Allias for in-place precomputed plans from FFTW or CUFFT. Used to compute inverse FFT.
"""
InversePlan{T} = Union{
    AbstractFFTs.ScaledPlan{Complex{T}, cFFTWPlan{Complex{T}, 1, true, 2, UnitRange{Int64}}, T},
    AbstractFFTs.ScaledPlan{Complex{T}, CUFFT.CuFFTPlan{Complex{T}, Complex{T}, 1, true, 2}, T},
    AbstractFFTs.ScaledPlan{Complex{T}, CUFFT.CuFFTPlan{Complex{T}, Complex{T}, 1, true, 2, 2, Nothing}, T}
} where {T<:AbstractFloat}