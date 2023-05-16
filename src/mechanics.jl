"""

    ice_load(c::PhysicalConstants{T}, H::T)

Compute ice load based on ice thickness.
"""
function ice_load(c::PhysicalConstants{T}, H::XMatrix) where {T<:AbstractFloat}
    return - c.rho_ice .* c.g .* H
end

#####################################################
# Precomputation
#####################################################

"""

    PrecomputedFastiso(dt, Omega, c, p)

Return a `struct` containing pre-computed tools to perform forward-stepping. Takes the
time step `dt`, the ComputationDomain `Omega`, the solid-Earth parameters `p` and 
physical constants `c` as input.
"""
function PrecomputedFastiso(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T};
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
    ODEsolver::Any = "ExplicitEuler",
    dt::T = T(years2seconds(1.0)),
    t_eta_snapshots::Vector{T} = [t_out[1], t_out[end]],
    eta_snapshots::Vector{<:XMatrix} = [p.effective_viscosity, p.effective_viscosity],
    u_viscous_0::Matrix{T} = copy(Omega.null),
    u_elastic_0::Matrix{T} = copy(Omega.null),
    active_geostate::Bool = false,
    verbose::Bool = true,
    kwargs...,
) where {T<:AbstractFloat}

    u_viscous_0, u_elastic_0 = kernelpromote(
        [u_viscous_0, u_elastic_0], Omega.arraykernel)   # Handle xPU architecture
    sstruct = init_superstruct(Omega, c, p, t_Hice_snapshots, Hice_snapshots,
        t_eta_snapshots, eta_snapshots, active_geostate; kwargs...)
    u, dudt, u_elastic, geoid, sealevel = forward_isostasy(
        dt, t_out, u_viscous_0, sstruct, ODEsolver, verbose)

    return FastIsoResults(
        t_out, sstruct.tools, u, dudt, u_elastic,
        geoid, sealevel, sstruct.Hice, sstruct.eta,
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

function init_superstruct(
    Omega::ComputationDomain{T},
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{Matrix{T}},
    t_eta_snapshots::Vector{T},
    eta_snapshots::Vector{<:XMatrix},
    active_geostate::Bool;
    geoid_0::Matrix{T} = copy(Omega.null),
    sealevel_0::Matrix{T} = copy(Omega.null),
    H_ice_ref::Matrix{T} = copy(Omega.null),
    H_water_ref::Matrix{T} = copy(Omega.null),
    b_ref::Matrix{T} = copy(Omega.null),
) where {T<:AbstractFloat}

    geoid_0, sealevel_0, H_ice_ref, H_water_ref, b_ref = kernelpromote(
        [geoid_0, sealevel_0, H_ice_ref, H_water_ref, b_ref], Omega.arraykernel)

    tools = PrecomputedFastiso(Omega, c, p)
    Hice = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Omega.arraykernel) )
    Hice_cpu = linear_interpolation( t_Hice_snapshots,
        kernelpromote(Hice_snapshots, Array) )
    eta = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Omega.arraykernel) )
    eta_cpu = linear_interpolation(t_eta_snapshots,
        kernelpromote(eta_snapshots, Array) )

    refgeostate = ReferenceGeoState(
        H_ice_ref, H_water_ref, b_ref,
        copy(sealevel_0),   # z0
        sealevel_0,         # sealevel
        T(0.0),             # sle_af
        T(0.0),             # V_pov
        T(0.0),             # V_den
        T(0.0),             # conservation_term
    )
    geostate = GeoState(
        Hice(0.0),                  # ice column
        copy(H_water_ref),          # water column
        copy(b_ref),                # bedrock position
        geoid_0,                    # geoid perturbation
        copy(sealevel_0),           # reference for external sl-forcing
        T(0.0), T(0.0), T(0.0),     # V_af terms
        T(0.0), T(0.0),             # V_pov terms
        T(0.0), T(0.0),             # V_den terms
        T(0.0),                     # total sl-contribution & conservation term
        0, years2seconds(10.0),     # countupdates, update step
    )
    return SuperStruct(Omega, c, p, tools, Hice, Hice_cpu, eta, eta_cpu,
        refgeostate, geostate, active_geostate)
end

function forward_isostasy(
    dt::T,
    t_out::Vector{T},
    u::XMatrix,
    sstruct::SuperStruct{T},
    ODEsolver::Any,
    verbose::Bool,
) where {T<:AbstractFloat}

    dudt = copy(u)
    u_out, dudt_out, u_el_out, geoid_out, sealevel_out = init_results(u, t_out)

    @inbounds for k in eachindex(t_out)[1:end]
        t0 = k == 1 ? T(0.0) : t_out[k-1]
        if verbose
            println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")
        end
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
        geoid_out[k] .= copy(kernelpromote(sstruct.geostate.geoid, Array))
        sealevel_out[k] .= copy(kernelpromote(sstruct.geostate.sealevel, Array))
    end
    return u_out, dudt_out, u_el_out, geoid_out, sealevel_out
end

function init_results(u, t_out)
    # initialize with placeholders
    placeholder = kernelpromote(u, Array)
    u_out = [copy(placeholder) for time in t_out]
    dudt_out = [copy(placeholder) for time in t_out]
    u_el_out = [copy(placeholder) for time in t_out]
    geoid_out = [copy(placeholder) for time in t_out]
    sealevel_out = [copy(placeholder) for time in t_out]
    return u_out, dudt_out, u_el_out, geoid_out, sealevel_out
end

function forwardstep_isostasy!(
    dudt::XMatrix,
    u::XMatrix,
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}
    apply_bc!(u, sstruct.Omega.N)
    if sstruct.active_geostate &&
        (t / sstruct.geostate.dt >= sstruct.geostate.countupdates)
        update_geostate!(sstruct, u, sstruct.Hice(t))
        sstruct.geostate.countupdates += 1
        # println("Updated GeoState at t=$(seconds2years(t))")
    end
    dudt_isostasy!(dudt, u, sstruct, t)
    return nothing
end

function dudt_isostasy!(
    dudt::XMatrix,
    u::XMatrix,
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}
    if sstruct.active_geostate
        load = ice_load(sstruct.c, sstruct.Hice(t))
    else
        load = get_loadchange(sstruct)
    end
    rhs = load - sstruct.tools.rhog .* u

    if sstruct.tools.negligible_gradD
        rhs += - sstruct.p.litho_rigidity .* real.( sstruct.tools.pifft *
            ( sstruct.Omega.biharmonic .* (sstruct.tools.pfft * u) ) )
    else
        dudxx = mixed_fdxx(u, sstruct.Omega.dx)
        dudyy = mixed_fdyy(u, sstruct.Omega.dy)
        Mxx = -sstruct.p.litho_rigidity .* (dudxx + sstruct.p.litho_poissonratio .* dudyy)
        Myy = -sstruct.p.litho_rigidity .* (dudyy + sstruct.p.litho_poissonratio .* dudxx)
        Mxy = -sstruct.p.litho_rigidity .* (1 - sstruct.p.litho_poissonratio) .*
            mixed_fdxy(u, sstruct.Omega.dx, sstruct.Omega.dy)
        rhs += mixed_fdxx(Mxx, sstruct.Omega.dx) + mixed_fdyy(Myy, sstruct.Omega.dy) +
            2 .* mixed_fdxy(Mxy, sstruct.Omega.dx, sstruct.Omega.dy)
    end

    dudt[:, :] .= real.(sstruct.tools.pifft * ((sstruct.tools.pfft * rhs) ./
        sstruct.Omega.pseudodiff)) ./ (2 .* sstruct.p.effective_viscosity)
    #     dudt[:, :] .= real.(tools.pifft * ((tools.pfft * (rhs ./ (2 .* p.effective_viscosity)) ) ./ Omega.pseudodiff))

    apply_bc!(dudt, sstruct.Omega.N)
    return nothing
end

function simple_euler!(
    u::XMatrix,
    dudt::XMatrix,
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
function apply_bc(u::XMatrix, N)
    u_bc = copy(u)
    return apply_bc!(u_bc, N)
end

function apply_bc!(u::XMatrix, N)
    CUDA.allowscalar() do
        u .-= (view(u, 1, 1) + view(u, 1, N) + view(u, N, 1) + view(u, N, N)) / 4
    end
end

# function apply_bc!(u::XMatrix, N) where {T<:AbstractFloat}
#     CUDA.allowscalar() do
#         u .-= sum(view(u, 1, :) + view(u, :, N) + view(u, N, :) + view(u, :, 1)) / (4*N)
#     end
# end

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
    load::XMatrix,
)
    return conv(load, tools.elasticgreen)[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end