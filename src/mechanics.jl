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
    u_viscous_0::Matrix{T} = copy(Omega.null),
    u_elastic_0::Matrix{T} = copy(Omega.null),
    interactive_geostate::Bool = false,
    verbose::Bool = true,
    kwargs...,
) where {T<:AbstractFloat}

    u_viscous_0, u_elastic_0 = kernelpromote(
        [u_viscous_0, u_elastic_0], Omega.arraykernel)   # Handle xPU architecture
    sstruct = SuperStruct(Omega, c, p, t_Hice_snapshots, Hice_snapshots,
        t_eta_snapshots, eta_snapshots, interactive_geostate; kwargs...)
    u, dudt, u_elastic, geoid, sealevel = forward_isostasy(
        dt, t_out, u_viscous_0, sstruct, ODEsolver, verbose)

    return FastisoResults(
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
    u_out, dudt_out, u_el_out, geoid_out, sealevel_out = init_results(u, t_out)

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
        
        u_out[k] .= copy(kernelpromote(u, Array))
        dudt_out[k] .= copy(kernelpromote(dudt, Array))
        geoid_out[k] .= copy(kernelpromote(sstruct.geostate.geoid, Array))
        sealevel_out[k] .= copy(kernelpromote(sstruct.geostate.sealevel, Array))
        u_el_out[k] .= samesize_conv(columnanom_load(sstruct),
            sstruct.tools.elasticgreen, sstruct.Omega)
    end
    return u_out, dudt_out, u_el_out, geoid_out, sealevel_out
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
    u_el_out = [copy(placeholder) for time in t_out]
    geoid_out = [copy(placeholder) for time in t_out]
    sealevel_out = [copy(placeholder) for time in t_out]
    return u_out, dudt_out, u_el_out, geoid_out, sealevel_out
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
    corner_bc!(u, sstruct.Omega.Nx, sstruct.Omega.Ny)

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

    coupled_viscoelastic = false

    if coupled_viscoelastic
        sstruct.geostate.dtloadanom .= columnanom_load(sstruct)
    end

    update_loadcolumns!(sstruct, u, sstruct.Hice(t))

    if coupled_viscoelastic
        sstruct.geostate.dtloadanom -= columnanom_load(sstruct)
        u += samesize_conv(sstruct.geostate.dtloadanom, sstruct.tools.elasticgreen,
            sstruct.Omega)
    end

    dudt_isostasy!(dudt, u, sstruct, t)
    return nothing
end

"""

    dudt_isostasy!()

Update the displacement rate `dudt`.
"""
function dudt_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    sstruct::SuperStruct{T},
    t::T,
) where {T<:AbstractFloat}

    rhs = - sstruct.c.g .* columnanom_full(sstruct)

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

    corner_bc!(dudt, sstruct.Omega.Nx, sstruct.Omega.Ny)
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
end

"""

    border_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at borders of domain is 0.
Same as Bueler et al. (2007).
"""
function border_bc(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    u_bc = copy(u)
    return border_bc!(u_bc, Nx, Ny)
end

function border_bc!(u::AbstractMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    allowscalar() do
        u .-= sum( view(u,1,:) + view(u,:,Nx) + view(u,Ny,:) + view(u,:,1) ) /
            (2*Nx + 2*Ny)
    end
end

#####################################################
# Elastic response
#####################################################
"""

    compute_elastic_response(Omega, tools, load)

For a computation domain `Omega`, compute the elastic response of the solid Earth
by convoluting the `load` with the Green's function stored in the pre-computed `tools`
(elements obtained from Farell 1972).
"""
function compute_elastic_response(
    Omega::ComputationDomain{T},
    tools::PrecomputedFastiso{T},
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return samesize_conv(load, tools.elasticgreen, Omega)
end