"""

    ice_load(c::PhysicalConstants{T}, H::T)

Compute ice load based on ice thickness.
"""
function ice_load(c::PhysicalConstants{T}, H::AbstractMatrix{T}) where {T<:AbstractFloat}
    return - c.ice_density .* c.g .* H
end

#####################################################
# Precomputation
#####################################################

"""

    PrecomputedFastiso(dt, Omega, p, c)

Return a `struct` containing pre-computed tools to perform forward-stepping. Takes the
time step `dt`, the ComputationDomain `Omega`, the solid-Earth parameters `p` and 
physical constants `c` as input.
"""
function PrecomputedFastiso(
    Omega::ComputationDomain{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T};
    quad_precision::Int = 4,
) where {T<:AbstractFloat}

    # Elastic response variables
    distance, greenintegrand_coeffs = get_greenintegrand_coeffs(T)
    greenintegrand_function = build_greenintegrand(distance, greenintegrand_coeffs)
    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    elasticgreen = get_elasticgreen(Omega, greenintegrand_function, quad_support, quad_coeffs)

    # Space-derivatives of rigidity
    D = kernelpromote(p.litho_rigidity, Array)
    Dx = mixed_fdx(D, Omega.dx)
    Dy = mixed_fdy(D, Omega.dy)
    Dxx = mixed_fdxx(D, Omega.dx)
    Dyy = mixed_fdyy(D, Omega.dy)
    Dxy = mixed_fdy( mixed_fdx(D, Omega.dx), Omega.dy )

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
        p1, p2 = CUDA.CUFFT.plan_fft(Xgpu), CUDA.CUFFT.plan_ifft(Xgpu)
        # Dx, Dy, Dxx, Dyy, Dxy, rhog = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy, rhog])
        Dx, Dy, Dxx, Dyy, Dxy = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy])
    else
        p1, p2 = plan_fft(Omega.X), plan_ifft(Omega.X)
    end

    rhog = p.mean_density .* c.g
    geoidgreen = get_geoidgreen(Omega, c)

    return PrecomputedFastiso(
        elasticgreen, fft(elasticgreen),
        p1, p2,
        Dx, Dy, Dxx, Dyy, Dxy, negligible_gradD,
        rhog,
        geoidgreen,
    )
end

#####################################################
# Forward integration
#####################################################

"""

forward_isostasy()

Main function.
List of all available solvers [here](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#OrdinaryDiffEq.jl-for-Non-Stiff-Equations).
"""
function forward_isostasy(
    t_out::Vector{T},
    Omega::ComputationDomain{T},
    tools::PrecomputedFastiso{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{Matrix{T}};
    t_eta_snapshots::Vector{T} = [t_out[1], t_out[end]],
    eta_snapshots::Vector{Matrix{T}} = [p.effective_viscosity, p.effective_viscosity],
    u_viscous_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    u_elastic_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    geoid_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    sealevel_0::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    H_ice_ref::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    H_water_ref::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    b_ref::Matrix{T} = fill(T(0.0), Omega.N, Omega.N),
    ODEsolver::Any = "SimpleEuler",
    dt::T = T(years2seconds(1.0)),
) where {T<:AbstractFloat}

    Hice = linear_interpolation( t_Hice_snapshots, Hice_snapshots )
    Hice_ode = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel) )
    eta = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Omega.arraykernel) )

    params = ODEParams(Omega, c, p, Hice_ode, tools)
    u_viscous_0 = kernelpromote(u_viscous_0, Omega.arraykernel)
    gs = GeoState(
        Hice(0.0), H_ice_ref,                   # ice column
        H_water_ref, H_water_ref,               # water column
        b_ref, b_ref,                           # bedrock position
        geoid_0,                                # geoid perturbation
        copy(sealevel_0),                       # reference for external sl-forcing
        sealevel_0, sealevel_0,                 # sealevel
        T(0.0), T(0.0), T(0.0), T(0.0),         # V_af terms
        T(0.0), T(0.0), T(0.0),                 # V_pov terms
        T(0.0), T(0.0), T(0.0),                 # V_den terms
        T(0.0), T(0.0),                         # total sl-contribution & conservation term
    )
    u, dudt, u_elastic, geoid, sealevel = solve_isostasy(
        t_out, u_viscous_0, gs, params, ODEsolver)

    return FastIsoResults(t_out, u, dudt, u_elastic, geoid, sealevel, Hice, eta)
end

function forward_isostasy(
    t_out::Vector{T},
    Omega::ComputationDomain{T},
    tools::PrecomputedFastiso{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T},
    Hice_snapshot::Matrix{T};
    kwargs...
) where {T<:AbstractFloat}
    t_Hice_snapshots = [t_out[1], t_out[end]]
    Hice_snapshots = [Hice_snapshot, Hice_snapshot]
    return forward_isostasy(
        t_out, Omega, tools, p, c, t_Hice_snapshots, Hice_snapshots)
end

function viscous_dudt!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    params::ODEParams{T},
    t::T,
) where {T<:AbstractFloat}

    Omega = params.Omega
    c = params.c
    p = params.p
    tools = params.tools
    Hice = params.Hice

    apply_bc!(u)
    uf = tools.pfft * u
    harmonic_uf = real.(tools.pifft * ( Omega.harmonic .* uf ))
    biharmonic_uf = real.( tools.pifft * ( Omega.biharmonic .* uf ) )

    term1 = ice_load(c, Hice(t))
    term2 = - tools.rhog .* u
    term3 = - p.litho_rigidity .* biharmonic_uf

    if tools.negligible_gradD
        rhs = term1 + term2 + term3
    else
        term4 = - T(2) .* tools.Dx .* mixed_fdx(harmonic_uf, Omega.dx)
        term5 = - T(2) .* tools.Dy .* mixed_fdy(harmonic_uf, Omega.dy)
        term6 = - real.(tools.pifft * (Omega.harmonic .* (tools.pfft * (p.litho_rigidity .* harmonic_uf))))
        term7 = tools.Dxx .* mixed_fdyy(u, Omega.dy)
        term8 = - T(2) .* tools.Dxy .* mixed_fdy( mixed_fdx(u, Omega.dx), Omega.dy )
        term9 = tools.Dyy .* mixed_fdxx(u, Omega.dx)

        rhs = term1 + term2 + term3 + 
                term4 + term5 + term6 + (T(1) - p.litho_poissonratio) .*
                (term7 + term8 + term9)
    end

    dudtf = (tools.pfft * rhs) ./ Omega.pseudodiff
    dudt[:, :] .= real.(tools.pifft * dudtf) ./ (T(2) .* p.effective_viscosity)
    apply_bc!(dudt)

    return nothing
end

function simple_euler!(
    u::AbstractMatrix{T},
    dudt::AbstractMatrix{T},
    dt::T,
) where {T<:AbstractFloat}
    u .+= dudt .* dt
    return nothing
end

function solve_isostasy(
    t_out::Vector{T},
    u::AbstractMatrix{T},
    gs::GeoState{T},
    params::ODEParams{T},
    ODEsolver::Any,
) where {T<:AbstractFloat}

    # initialize with placeholders
    placeholder = kernelpromote(u, Array)
    u_out = [copy(placeholder) for time in t_out]
    dudt_out = [copy(placeholder) for time in t_out]
    u_el_out = [copy(placeholder) for time in t_out]
    geoid_out = [copy(placeholder) for time in t_out]
    sealevel_out = [copy(placeholder) for time in t_out]

    dudt = copy(u)
    dt = years2seconds( T(1.0) )

    for k in eachindex(t_out)[1:end]
        t0 = k == 1 ? T(0.0) : t_out[k-1]
        println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")

        if isa(ODEsolver, OrdinaryDiffEqAlgorithm)
            prob = ODEProblem(viscous_dudt!, u, (t0, t_out[k]), params)
            sol = solve(prob, ODEsolver, reltol = 1e-3)

            u .= sol(t_out[k], Val{0})
            dudt .= sol(t_out[k], Val{1})
        else
            for t in t0:dt:t_out[k]
                viscous_dudt!(dudt, u, params, t)
                simple_euler!(u, dudt, dt)
            end
        end

        u_out[k] .= kernelpromote(u, Array)
        dudt_out[k] .= kernelpromote(dudt, Array)

        update_geostate!(gs, u_out[k], params.Hice(t_out[k]),
            params.Omega, params.c, params.p, params.tools)
        geoid_out[k] .= copy(gs.geoid)
        sealevel_out[k] .= copy(gs.sealevel)
    end
    return u_out, dudt_out, u_el_out, geoid_out, sealevel_out
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
    return conv(load, tools.elasticgreen)[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end