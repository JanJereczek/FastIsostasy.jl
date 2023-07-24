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
    eta_snapshots::Vector{<:AbstractMatrix{T}} = [p.effective_viscosity, p.effective_viscosity],
    u_0::Matrix{T} = copy(Omega.null),
    ue_0::Matrix{T} = copy(Omega.null),
    interactive_geostate::Bool = false,
    verbose::Bool = true,
    kwargs...,
) where {T<:AbstractFloat}

    u_0, ue_0 = kernelpromote([u_0, ue_0], Omega.arraykernel)   # Handle xPU architecture
    sstruct = SuperStruct(Omega, c, p, t_Hice_snapshots, Hice_snapshots,
        t_eta_snapshots, eta_snapshots, interactive_geostate; kwargs...)
    u, dudt, ue, geoid, sealevel = forward_isostasy(
        dt, t_out, u_0, sstruct, ODEsolver, verbose)

    return FastisoResults(
        t_out, sstruct.tools, u, dudt, ue,
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

"""

    forward_isostasy()

Forward-integrate the isostatic adjustment.
"""
function forward_isostasy(
    dt::T,
    t_out::Vector{T},
    u::AbstractMatrix{T},
    sstruct::SuperStruct{T},
    ODEsolver::Any,
    verbose::Bool,
) where {T<:AbstractFloat}

    dudt = copy(u)
    u_out, dudt_out, ue_out, geoid_out, sealevel_out = init_results(u, t_out)

    @inbounds for k in eachindex(t_out)[1:end]
        t0 = k == 1 ? T(0.0) : t_out[k-1]
        if verbose
            println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")
        end
        if isa(ODEsolver, OrdinaryDiffEqAlgorithm)
            prob = ODEProblem(forwardstep_isostasy!, u, (t0, t_out[k]), sstruct)
            
            if ODEsolver == Euler()
                sol = solve(prob, ODEsolver, reltol=1e-3, dt = dt)
            else
                sol = solve(prob, ODEsolver, reltol=1e-3) #, dtmin = years2seconds(0.1))
            end

            u .= sol(t_out[k], Val{0})
            dudt .= sol(t_out[k], Val{1})
        else
            for t in t0:dt:t_out[k]
                forwardstep_isostasy!(dudt, u, sstruct, t)
                explicit_euler!(u, dudt, dt)
            end
        end
        
        # Convert outputs back to CPU for postprocessing
        u_out[k] .= copy(kernelpromote(u, Array))
        dudt_out[k] .= copy(kernelpromote(dudt, Array))
        geoid_out[k] .= copy(kernelpromote(sstruct.geostate.geoid, Array))
        sealevel_out[k] .= copy(kernelpromote(sstruct.geostate.sealevel, Array))
        ue_out[k] .= copy(kernelpromote(sstruct.geostate.ue, Array))
    end
    return u_out, dudt_out, ue_out, geoid_out, sealevel_out
end

"""

    init_results()

Initialize some `Vector{Matrix}` where results shall be later stored.
"""
function init_results(u, t_out)
    # initialize with placeholders
    placeholder = kernelpromote(u, Array)
    u_out = [copy(placeholder) for time in t_out]
    dudt_out = [copy(placeholder) for time in t_out]
    ue_out = [copy(placeholder) for time in t_out]
    geoid_out = [copy(placeholder) for time in t_out]
    sealevel_out = [copy(placeholder) for time in t_out]
    return u_out, dudt_out, ue_out, geoid_out, sealevel_out
end

"""

    forwardstep_isostasy!(dudt, u, sstruct, t)

Forward integrate the isostatic adjustment over a single time step by updating the
displacement rate `dudt`, and the sea-level state contained within `sstruct::SuperStruct`.
"""
function forwardstep_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}

    sstruct.Omega.bc!(u, sstruct.Omega.Nx, sstruct.Omega.Ny)
    update_bedrock!(sstruct, u)

    update_loadcolumns!(sstruct, sstruct.Hice(t))
    # Only update the geoid and sea level if geostate is interactive.
    # As integration requires smaller time steps than diagnostics,
    # only update geostate every sstruct.geostate.dt
    if sstruct.interactive_geostate &&
        (t / sstruct.geostate.dt >= sstruct.geostate.countupdates)
        update_geoid!(sstruct)
        update_sealevel!(sstruct)
        sstruct.geostate.countupdates += 1
        # println("Updated GeoState at t=$(seconds2years(t))")
    end

    update_elasticresponse!(sstruct)
    dudt_isostasy!(dudt, u, sstruct, t)
    return nothing
end

"""

    dudt_isostasy!()

Update the displacement rate `dudt` of the viscous response.
"""
function dudt_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}

    Omega = sstruct.Omega
    rhs = - sstruct.c.g .* columnanom_full(sstruct)

    if sstruct.tools.negligible_gradD
        biharmonic_u = Omega.biharmonic .* (sstruct.tools.pfft * u)
        rhs += - sstruct.p.litho_rigidity .* real.( sstruct.tools.pifft * biharmonic_u )
    else
        dudxx, dudyy = Omega.fdxx(u), Omega.fdyy(u)
        Mxx = -sstruct.p.litho_rigidity .* (dudxx + sstruct.p.litho_poissonratio .* dudyy)
        Myy = -sstruct.p.litho_rigidity .* (dudyy + sstruct.p.litho_poissonratio .* dudxx)
        Mxy = -sstruct.p.litho_rigidity .* (1 - sstruct.p.litho_poissonratio) .* Omega.fdxy(u)
        rhs += Omega.fdxx(Mxx) + Omega.fdyy(Myy) + 2 .* Omega.fdxy(Mxy)
    end

    # dudt[:, :] .= real.(sstruct.tools.pifft * ((sstruct.tools.pfft * rhs) ./
    #     Omega.pseudodiff)) ./ (2 .* sstruct.p.effective_viscosity)
    dudt[:, :] .= real.(sstruct.tools.pifft * ((sstruct.tools.pfft * (rhs ./ 
        (2 .* sstruct.p.effective_viscosity)) ) ./ Omega.pseudodiff))

    return nothing
end

"""

    explicit_euler!()

Update the state `u` by performing an explicit Euler integration of its derivative `dudt`
over a time step `dt`.
"""
function explicit_euler!(
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

    corner_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at corners of domain is 0.
"""
function corner_bc(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    u_bc = copy(u)
    return corner_bc!(u_bc, Nx, Ny)
end

function corner_bc!(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    allowscalar() do
        u .-= ( view(u,1,1) + view(u,1,Nx) + view(u,Ny,1) + view(u,Ny,Nx) ) / 4
    end
    return u
end

"""

    edge_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at borders of domain is 0.
Same as Bueler et al. (2007).
"""
function edge_bc(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    u_bc = copy(u)
    return edge_bc!(u_bc, Nx, Ny)
end

function edge_bc!(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    allowscalar() do
        u .-= sum( view(u,1,:) + view(u,:,Nx) + view(u,Ny,:) + view(u,:,1) ) /
            (2*Nx + 2*Ny)
    end
    return u
end

no_bc!(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int) = nothing
no_bc(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int) = u

function no_mean_bc!(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    u .= u .- mean(u)
    return u
end

function no_mean_bc(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    u_bc = copy(u)
    return no_mean_bc!(u_bc, Nx, Ny)
end

#####################################################
# Elastic response
#####################################################
"""

    update_elasticresponse!(sstruct::SuperStruct)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from (Farell 1972), see [PrecomputedFastiso](@ref).
"""
function update_elasticresponse!(sstruct::SuperStruct{<:AbstractFloat})
    rgh = correct_surfacedisctortion(columnanom_load(sstruct), sstruct)
    sstruct.geostate.ue .= samesize_conv(sstruct.tools.elasticgreen,
        rgh, sstruct.Omega, no_bc)
    return nothing
end