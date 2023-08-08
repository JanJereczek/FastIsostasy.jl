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
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    Hice_snapshot::AbstractMatrix{T};
    kwargs...,
) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    t_Hice_snapshots = [t_out[1], t_out[end]]
    Hice_snapshots = [Hice_snapshot, Hice_snapshot]
    return fastisostasy(t_out, Omega, c, p, t_Hice_snapshots, Hice_snapshots; kwargs...)
end

function fastisostasy(
    t_out::Vector{T},
    Omega::ComputationDomain{T, M},
    c::PhysicalConstants{T},
    p::LateralVariability{T, M},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:AbstractMatrix{T}};
    dt::T = T(years2seconds(1.0)),
    ODEsolver::Any = "ExplicitEuler",
    t_eta_snapshots::Vector{T} = [t_out[1], t_out[end]],
    eta_snapshots::Vector{<:AbstractMatrix{T}} = [p.effective_viscosity, p.effective_viscosity],
    interactive_geostate::Bool = false,
    verbose::Bool = true,
    kwargs...,
) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    fi = FastIso(Omega, c, p, t_out, interactive_geostate, 
        t_Hice_snapshots, Hice_snapshots,
        t_eta_snapshots, eta_snapshots; kwargs...)
    forward_isostasy!(fi, dt, ODEsolver, verbose)
    return FastIsoResults(fi)
end

"""
    forward_isostasy!()

Forward-integrate the isostatic adjustment.
"""
# function forward_isostasy!(fi::FastIso{T, M}, dt::T, ODEsolver::Any,
#     verbose::Bool) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
#     t_out = fi.t_out
#     forward_isostasy!(fi, t_out, dt, ODEsolver, verbose)
#     return nothing
# end

function forward_isostasy!(fi::FastIso{T, M}, dt::T, ODEsolver::Any,
    verbose::Bool) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    t1 = time()
    t_out = fi.t_out

    # Make a first diagnotisc update to store these values.
    update_diagnostics!(fi.geostate.dudt, fi.geostate.u, fi, 0.0)
    write_out!(fi, 1)

    # Check if solver available and initialize ODEProblem if possible.
    if !isa(ODEsolver, OrdinaryDiffEqAlgorithm)
        error("Provided ODEsolver is not supported.")
    elseif ODEsolver != Euler()
        diffeq = (alg = ODEsolver, abstol = 1e-9, reltol = 1e-6)
        fi_ode = CoupledODEs(update_diagnostics!, fi.geostate.u, fi; diffeq)
        # prob = ODEProblem(update_diagnostics!, u, (t_out[1], t_out[2]), fi)
    end

    @inbounds for k in eachindex(t_out)[2:end]
        difft = t_out[k]-t_out[k-1]
        X, t = trajectory(fi_ode, difft, fi.geostate.u; t0 = t_out[k-1], Δt = difft)
        fi.geostate.u .= reshape(X[2, :], fi.Omega.Nx, fi.Omega.Ny)
        update_diagnostics!(fi.geostate.dudt, fi.geostate.u, fi, t[2])
        println("$k, $(t_out[k]), $(minimum(fi.geostate.u))")
        # (t0+Ttr):Δt:(t0+Ttr+T)
        write_out!(fi, k)
    end

    # @inbounds for k in eachindex(t_out)[2:end]
    #     if verbose
    #         println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")
    #     end

    #     if ODEsolver == Euler()
    #         @inbounds for t in t_out[k-1]:dt:t_out[k]
    #             update_diagnostics!(dudt, u, fi, t)
    #             explicit_euler!(u, dudt, dt)
    #         end
    #     else

            

    #         # prob = remake(prob, u0 = fi.geostate.u, tspan = (t_out[k-1], t_out[k]), p = fi)
    #         # sol = solve(prob, ODEsolver, reltol=1e-3)
    #         # fi.geostate.dudt = sol(t_out[k], Val{1})
    #     end

    # end

    fi.computation_time += time() - t1
    return nothing
end

"""
    update_diagnostics!(dudt, u, fi, t)

Update all the diagnotisc variables, i.e. all fields of `fi.geostate` apart
from the displacement, which requires an integrator.
"""
function update_diagnostics!(dudt::M, u::M, fi::FastIso{T, M}, t::T,
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    # Order really matters here!
    if fi.internal_loadupdate
        update_loadcolumns!(fi, fi.tools.Hice(t))
    end
    update_elasticresponse!(fi)
    fi.Omega.bc!(u, fi.Omega.Nx, fi.Omega.Ny)
    update_bedrock!(fi, u)
    
    # Only update the geoid and sea level if geostate is interactive.
    # As integration requires smaller time steps than diagnostics,
    # only update geostate every fi.geostate.dt
    if fi.interactive_geostate &&
        (t / fi.geostate.dt >= fi.geostate.countupdates)
        update_geoid!(fi)
        update_sealevel!(fi)
        fi.geostate.countupdates += 1
        # println("Updated GeoState at t=$(seconds2years(t))")
    end

    dudt_isostasy!(dudt, u, fi, t)
    return nothing
end

"""
    dudt_isostasy!()

Update the displacement rate `dudt` of the viscous response.
"""
function dudt_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    fi::FastIso{T, M},
    t::T,
) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    Omega = fi.Omega
    rhs = - fi.c.g .* columnanom_full(fi)

    if fi.tools.negligible_gradD
        biharmonic_u = Omega.biharmonic .* (fi.tools.pfft * u)
        rhs += - fi.p.litho_rigidity .* real.( fi.tools.pifft * biharmonic_u )
    else
        dudxx, dudyy = Omega.fdxx(u), Omega.fdyy(u)
        Mxx = -fi.p.litho_rigidity .* (dudxx + fi.p.litho_poissonratio .* dudyy)
        Myy = -fi.p.litho_rigidity .* (dudyy + fi.p.litho_poissonratio .* dudxx)
        Mxy = -fi.p.litho_rigidity .* (1 - fi.p.litho_poissonratio) .* Omega.fdxy(u)
        rhs += Omega.fdxx(Mxx) + Omega.fdyy(Myy) + 2 .* Omega.fdxy(Mxy)
    end

    # dudt[:, :] .= real.(fi.tools.pifft * ((fi.tools.pfft * rhs) ./
    #     Omega.pseudodiff)) ./ (2 .* fi.p.effective_viscosity)
    dudt[:, :] .= real.(fi.tools.pifft * ((fi.tools.pfft * (rhs ./ 
        (2 .* fi.p.effective_viscosity)) ) ./ Omega.pseudodiff))

    return dudt
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
        u .-= ( view(u,1,1) + view(u,Nx,1) + view(u,1,Ny) + view(u,Nx,Ny) ) / 4
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
        u .-= sum( view(u,1,:) + view(u,Nx,:) + view(u,:,Ny) + view(u,:,1) ) /
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
    update_elasticresponse!(fi::FastIso)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from (Farell 1972), see [PrecomputedFastiso](@ref).
"""
function update_elasticresponse!(fi::FastIso{T, M}
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    rgh = correct_surfacedisctortion(columnanom_load(fi), fi)
    fi.geostate.ue .= samesize_conv(fi.tools.elasticgreen, rgh, fi.Omega, no_bc)
    return nothing
end