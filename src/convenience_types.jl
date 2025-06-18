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

"""
    AbstractLithosphere

Available subtypes are:
- [`LaterallyConstantLithosphere`](@ref)
- [`LaterallyVariableLithosphere`](@ref)
"""
abstract type AbstractLithosphere end

"""
    AbstractMantle

Available subtypes are:
- [`LaterallyConstantMantle`](@ref)
- [`LaterallyVariableMantle`](@ref)
"""
abstract type AbstractMantle end

"""
    AbstractRheology

Available subtypes are:
- [`RelaxedRheology`](@ref)
- [`ViscousRheology`](@ref)
"""
abstract type AbstractRheology end

"""
    LaterallyConstantLithosphere

Assume a laterally constant lithospheric thickness (and rigidity) across the domain.
This generally improves the performance of the solver, but is less realistic.
"""
struct LaterallyConstantLithosphere <: AbstractLithosphere end

"""
    LaterallyVariableLithosphere

Assume a laterally variable lithospheric thickness (and rigidity) across the domain.
This generally improves the realism of the model, but is more computationally expensive.
"""
struct LaterallyVariableLithosphere <: AbstractLithosphere end

"""
    LaterallyConstantMantle

Assume a laterally-constant mantle properties (relaxation time or viscosity
depending on the [`AbstractRheology`](@ref) that is used) across the domain.
This generally improves the performance of the solver, but is less realistic.
"""
struct LaterallyConstantMantle <: AbstractMantle end

"""
    LaterallyVariableMantle

Assume a laterally-variable mantle properties (relaxation time or viscosity
depending on the [`AbstractRheology`](@ref) that is used) across the domain.
This generally improves the realism of the model, but is more computationally expensive.
"""
struct LaterallyVariableMantle <: AbstractMantle end

"""
    RelaxedRheology

Assume a relaxed rheology in the mantle, governed by a relaxation time.
This generally offers worse performance and is less realistic than a viscous rheology.
It is only included for legacy purpose (e.g. comparison among solvers).
"""
struct RelaxedRheology <: AbstractRheology end

"""
    ViscousRheology

Assume a viscous rheology in the mantle, governed by a viscosity.
This is the most realistic rheology and generally offers the best performance.
It is the default rheology used in the solver.
"""
struct ViscousRheology <: AbstractRheology end

struct EarthModel{L<:AbstractLithosphere, M<:AbstractMantle, R<:AbstractRheology}
    lithosphere::L      # lc or lv
    mantle::M           # lc or lv
    rheology::R         # relaxed or viscous
end

# lc, lc, relaxed = ELRA
# lv, lv, relaxed = LVELRA
# lc, lc, viscous = ELVA
# lv, lv, viscous = LVELVA

# But we could also do:
# lc, lv, relaxed => ELRA solver
# lv, lc, relaxed => LVELRA solver
# lc, lv, viscous => ELVA solver
# lv, lc, viscous => LVELVA solver