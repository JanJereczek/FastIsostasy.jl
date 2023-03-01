"""

    FastIsoResults

A mutable struct containing the results of FastIsostasy:
    - `t_out` the time output vector
    - `u3D_elastic` the elastic response over `t_out`
    - `u3D_viscous` the viscous response over `t_out`
    - `dudt3D_viscous` the displacement rate over `t_out`
    - `geoid3D` the geoid response over `t_out`
    - `Hice` an interpolator of the ice thickness over time
    - `eta` an interpolator of the upper-mantle viscosity over time
"""
struct FastIsoResults{T<:AbstractFloat}
    t_out::Vector{T}
    viscous::Vector{Matrix{T}}
    displacement_rate::Vector{Matrix{T}}
    elastic::Vector{Matrix{T}}
    geoid::Vector{Matrix{T}}
    Hice::Interpolations.Extrapolation
    eta::Interpolations.Extrapolation
end

"""

    ice_load(c::PhysicalConstants{T}, H::T)

Compute ice load based on ice thickness.
"""
function ice_load(c::PhysicalConstants{T}, H::Matrix{T}) where {T<:AbstractFloat}
    return -c.ice_density .* c.g .* H
end

#####################################################
# Precomputation
#####################################################

"""

    precompute_fastiso(dt, Omega, p, c)

Return a `struct` containing pre-computed tools to perform forward-stepping. Takes the
time step `dt`, the ComputationDomain `Omega`, the solid-Earth parameters `p` and 
physical constants `c` as input.
"""
function precompute_fastiso(
    Omega::ComputationDomain{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T};
    quad_precision::Int = 4,
) where {T<:AbstractFloat}

    # Elastic response
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    loadresponse = get_integrated_loadresponse(Omega, quad_support, quad_coeffs)

    # Space-derivatives of rigidity 
    Dx = mixed_fdx(p.litho_rigidity, Omega.dx)
    Dy = mixed_fdy(p.litho_rigidity, Omega.dy)
    Dxx = mixed_fdxx(p.litho_rigidity, Omega.dx)
    Dyy = mixed_fdyy(p.litho_rigidity, Omega.dy)
    Dxy = mixed_fdy( mixed_fdx(p.litho_rigidity, Omega.dx), Omega.dy )

    omega_zeros = fill(T(0.0), Omega.N, Omega.N)
    zero_tol = 1e-2
    negligible_gradD = isapprox(Dx, omega_zeros, atol = zero_tol) &
                        isapprox(Dy, omega_zeros, atol = zero_tol) &
                        isapprox(Dxx, omega_zeros, atol = zero_tol) &
                        isapprox(Dyy, omega_zeros, atol = zero_tol) &
                        isapprox(Dxy, omega_zeros, atol = zero_tol)
    
    # FFT plans depening on CPU vs. GPU usage
    if Omega.use_cuda
        Xgpu = CuArray(Omega.X)
        p1 = CUDA.CUFFT.plan_fft(Xgpu)
        p2 = CUDA.CUFFT.plan_ifft(Xgpu)
        # Dx, Dy, Dxx, Dyy, Dxy, rhog = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy, rhog])
        Dx, Dy, Dxx, Dyy, Dxy = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy])
    else
        p1, p2 = plan_fft(Omega.X), plan_ifft(Omega.X)
    end

    rhog = p.mean_density .* c.g
    geoid_green = get_geoid_green(Omega.Θ, c)

    return PrecomputedFastiso(
        loadresponse, fft(loadresponse),
        p1, p2,
        Dx, Dy, Dxx, Dyy, Dxy, negligible_gradD,
        rhog,
        geoid_green,
    )
end

#####################################################
# Forward integration
#####################################################

function isostasy(
    t_out::Vector{T},
    Omega::ComputationDomain{T},
    tools::PrecomputedFastiso{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{Matrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{Matrix{T}};
    u_viscous_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    u_elastic_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    geoid_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
) where {T<:AbstractFloat}

    Hice = linear_interpolation(t_Hice_snapshots, Hice_snapshots)
    eta = linear_interpolation(t_eta_snapshots, eta_snapshots)
    sol, dudt = compute_viscous_response(t_out, u_viscous_0, Omega, Hice, tools, p, c)

    u_elastic = [u_elastic_0 for time in t_out]
    geoid = [geoid_0 for time in t_out]
    for i in eachindex(t_out)[1:end]
        t = t_out[i]
        u_elastic[i] .+= compute_elastic_response(Omega, tools, -c.ice_density.*Hice(t))
        # update_columnchanges!(lc, sol.u[i], Hice(t))
        # geoid[i] .+= compute_geoid_response(c, p, Omega, tools, lc)
    end

    return FastIsoResults(t_out, sol.u, dudt, u_elastic, geoid, Hice, eta)
end

function compute_viscous_response(
    t_out::Vector{T},
    u_viscous::AbstractMatrix{T},
    Omega::ComputationDomain,
    Hice::Interpolations.Extrapolation,
    tools::PrecomputedFastiso,
    p::MultilayerEarth,
    c::PhysicalConstants,
) where {T<:AbstractFloat}

    params = ODEParams(Omega, c, p, Hice, tools)
    prob = ODEProblem(f!, u_viscous, extrema(t_out), params)
    sol = solve(prob, BS3(), saveat = t_out)
    dudt = [sol(t, Val{1}) for t in t_out]
    return sol, dudt
end

function f!(du::Matrix{T}, u::Matrix{T}, params::ODEParams{T}, t::T) where {T<:AbstractFloat}
    # println("t = $(Int(round(seconds2years(t)))) years...")
    Omega = params.Omega
    c = params.c
    p = params.p
    tools = params.tools
    Hice = params.Hice

    apply_bc!(u)
    uf = tools.pfft * u
    harmonic_uf = real.(tools.pifft * ( Omega.harmonic_coeffs .* uf ))
    biharmonic_uf = real.( tools.pifft * ( Omega.biharmonic_coeffs .* uf ) )

    term1 = ice_load(c, Hice(t))
    term2 = - tools.rhog .* u
    term3 = - p.litho_rigidity .* biharmonic_uf
    term6 = - real.(tools.pifft * (Omega.harmonic_coeffs .* (tools.pfft * (p.litho_rigidity .* harmonic_uf))))

    if tools.negligible_gradD
        rhs = term1 + term2 + term3 + term6
    else
        term4 = - T(2) .* tools.Dx .* mixed_fdx(harmonic_uf, Omega.dx)
        term5 = - T(2) .* tools.Dy .* mixed_fdy(harmonic_uf, Omega.dy)
        term7 = tools.Dxx .* mixed_fdyy(u, Omega.dy)
        term8 = - T(2) .* tools.Dxy .* mixed_fdy( mixed_fdx(u, Omega.dx), Omega.dy )
        term9 = tools.Dyy .* mixed_fdxx(u, Omega.dx)

        rhs = term1 + term2 + term3 + 
                term4 + term5 + term6 + (T(1) - p.litho_poissonratio) .*
                (term7 + term8 + term9)
    end

    dudtf = (tools.pfft * rhs) ./ Omega.pseudodiff_coeffs
    du[:, :] .= real.(tools.pifft * dudtf) ./ (T(2) .* p.effective_viscosity)
    apply_bc!(du)

    return nothing
end

#####################################################
# BCs
#####################################################

"""

    apply_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at corners of domain is 0.
Whereas Bueler et al. (2007) take the edges for this computation, we take the corners
because they represent the far-field better.
"""
function apply_bc(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    u_bc = copy(u)
    return apply_bc!(u_bc)
end

function apply_bc!(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    CUDA.allowscalar() do
        u .-= (u[1,1] + u[1,end] + u[end,1] + u[end,end]) / T(4)
    end
end

#####################################################
# Elastic response
#####################################################
"""

    compute_elastic_response(tools, load)

For a computation domain `Omega`, compute the elastic response of the solid Earth
by convoluting the `load` with the Green's function stored in the pre-computed `tools`
(elements obtained from Farell 1972).
"""
function compute_elastic_response(
    Omega::ComputationDomain,
    tools::PrecomputedFastiso,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return conv(load, tools.loadresponse)[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end