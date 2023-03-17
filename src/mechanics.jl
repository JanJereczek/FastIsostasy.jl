"""

    ice_load(c::PhysicalConstants{T}, H::T)

Compute ice load based on ice thickness.
"""
function ice_load(c::PhysicalConstants{T}, H::AbstractMatrix{T}) where {T<:AbstractFloat}
    return - c.rho_ice .* c.g .* H
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
        kernelpromote(geoidgreen, Omega.arraykernel),
    )
end

#####################################################
# Forward integration
#####################################################

"""

    fastisostasy()

Main function.
List of all available solvers [here](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#OrdinaryDiffEq.jl-for-Non-Stiff-Equations).
"""
function fastisostasy(
    t_out::Vector{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{Matrix{T}};
    t_eta_snapshots::Vector{T} = [t_out[1], t_out[end]],
    eta_snapshots::Vector{<:AbstractMatrix{T}} = [p.effective_viscosity, p.effective_viscosity],
    u_viscous_0::Matrix{T} = copy(Omega.null),
    # u_elastic_0::Matrix{T} = copy(Omega.null),
    geoid_0::Matrix{T} = copy(Omega.null),
    sealevel_0::Matrix{T} = copy(Omega.null),
    H_ice_ref::Matrix{T} = copy(Omega.null),
    H_water_ref::Matrix{T} = copy(Omega.null),
    b_ref::Matrix{T} = copy(Omega.null),
    ODEsolver::Any = "ExplicitEuler",
    dt::T = T(years2seconds(1.0)),
    active_geostate::Bool = true,
) where {T<:AbstractFloat}
    u_viscous_0, geoid_0, sealevel_0, H_ice_ref, H_water_ref, b_ref = kernelpromote(
        [u_viscous_0, geoid_0, sealevel_0, H_ice_ref, H_water_ref, b_ref],
        Omega.arraykernel,
    )
    tools = PrecomputedFastiso(Omega, p, c)
    Hice = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel) )
    Hice_cpu = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Array) )
    eta = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Omega.arraykernel) )
    u_viscous_0 = kernelpromote(u_viscous_0, Omega.arraykernel)

    geostate = GeoState(
        Hice(0.0), H_ice_ref,                   # ice column
        copy(H_water_ref), H_water_ref,         # water column
        copy(b_ref), b_ref,                     # bedrock position
        geoid_0,                                # geoid perturbation
        copy(sealevel_0),                       # reference for external sl-forcing
        copy(sealevel_0), sealevel_0,           # sealevel
        T(0.0), T(0.0), T(0.0), T(0.0),         # V_af terms
        T(0.0), T(0.0), T(0.0),                 # V_pov terms
        T(0.0), T(0.0), T(0.0),                 # V_den terms
        T(0.0), T(0.0),                         # total sl-contribution & conservation term
    )
    sstruct = SuperStruct(Omega, c, p, Hice, Hice_cpu, tools, geostate, active_geostate)

    u, dudt, u_elastic, geoid, sealevel = forward_isostasy(
        dt, t_out, u_viscous_0, geostate, sstruct, ODEsolver)

    return FastIsoResults(
        t_out, tools, u, dudt, u_elastic,
        geoid, sealevel, Hice, eta,
    )
end

function fastisostasy(
    t_out::Vector{T},
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    Hice_snapshot::Matrix{T};
    kwargs...,
) where {T<:AbstractFloat}
    t_Hice_snapshots = [t_out[1], t_out[end]]
    Hice_snapshots = [Hice_snapshot, Hice_snapshot]
    return fastisostasy(t_out, Omega, c, p, t_Hice_snapshots, Hice_snapshots; kwargs...)
end

function forward_isostasy(
    dt::T,
    t_out::Vector{T},
    u::AbstractMatrix{T},
    geostate::GeoState{T},
    sstruct::SuperStruct{T},
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

    for k in eachindex(t_out)[1:end]
        t0 = k == 1 ? T(0.0) : t_out[k-1]
        println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")

        if isa(ODEsolver, OrdinaryDiffEqAlgorithm)
            prob = ODEProblem(forwardstep_isostasy!, u, (t0, t_out[k]), sstruct)
            sol = solve(prob, ODEsolver, reltol=1e-3) #, dtmin = years2seconds(0.1)), , dt=dt

            u .= sol(t_out[k], Val{0})
            dudt .= sol(t_out[k], Val{1})
        else
            for t in t0:dt:t_out[k]
                forwardstep_isostasy!(dudt, u, sstruct, t)
                simple_euler!(u, dudt, dt)
            end
        end
        
        u_out[k] .= copy(kernelpromote(u, Array))
        dudt_out[k] .= copy(kernelpromote(dudt, Array))
        geoid_out[k] .= copy(kernelpromote(geostate.geoid, Array))
        sealevel_out[k] .= copy(kernelpromote(geostate.sealevel, Array))
    end
    return u_out, dudt_out, u_el_out, geoid_out, sealevel_out
end

function forwardstep_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}
    apply_bc!(u)
    if sstruct.active_geostate
        update_geostate!(
            sstruct.geostate, u, sstruct.Hice(t),
            sstruct.Omega, sstruct.c, sstruct.p, sstruct.tools,
        )
    end
    dudt_isostasy!(dudt, u, sstruct, t)
    return nothing
end

function dudt_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}
    Omega = sstruct.Omega
    c = sstruct.c
    p = sstruct.p
    tools = sstruct.tools

    uf = tools.pfft * u
    harmonic_u = real.(tools.pifft * ( Omega.harmonic .* uf ))
    biharmonic_u = real.( tools.pifft * ( Omega.biharmonic .* uf ) )

    if sstruct.active_geostate
        term1 = get_loadchange(sstruct.geostate, Omega, c)
    else
        term1 = ice_load(c, sstruct.Hice(t))
    end

    term2 = - tools.rhog .* u
    term3 = - p.litho_rigidity .* biharmonic_u

    if tools.negligible_gradD
        rhs = term1 + term2 + term3
    else
        term4 = - T(2) .* tools.Dx .* mixed_fdx(harmonic_u, Omega.dx)
        term5 = - T(2) .* tools.Dy .* mixed_fdy(harmonic_u, Omega.dy)
        term6 = - real.(tools.pifft * (Omega.harmonic .* (tools.pfft * (p.litho_rigidity .* harmonic_u))))
        term7 = tools.Dxx .* mixed_fdyy(u, Omega.dy)
        term8 = - T(2) .* tools.Dxy .* mixed_fdy( mixed_fdx(u, Omega.dx), Omega.dy )
        term9 = tools.Dyy .* mixed_fdxx(u, Omega.dx)

        rhs = term1 + term2 + term3 + 
                term4 + term5 + term6 + (T(1) - p.litho_poissonratio) .*
                (term7 + term8 + term9)
    end

    dudtf = (tools.pfft * rhs) ./ Omega.pseudodiff
    dudt[:, :] .= T.( real.(tools.pifft * dudtf) ./ (2 .* p.effective_viscosity) )
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

function welcome_user()
    println("Welcome to FastIso")
end