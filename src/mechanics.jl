#####################################################
# Forward integration
#####################################################
"""
    solve!(fip)

Solve the isostatic adjustment problem defined in `fip::FastIsoProblem`.
"""
function solve!(fip::FastIsoProblem{T, M}) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    t1 = time()
    if !(fip.internal_loadupdate)
        error("`solve!` does not support external updating of the load. Use `step!` instead.")
    end

    t_out = fip.out.t

    # Make a first diagnotisc update to store these values. (k=1)
    update_diagnostics!(fip.geostate.dudt, fip.geostate.u, fip, t_out[1])
    write_out!(fip, 1)

    # Initialize dummy ODEProblem and perform integration.
    dummy = ODEProblem(update_diagnostics!, fip.geostate.u, (0.0, 1.0), fip)
    @inbounds for k in eachindex(t_out)[2:end]
        if fip.verbose
            println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")
        end
        if fip.diffeq.alg != SimpleEuler()
            prob = remake(dummy, u0 = fip.geostate.u, tspan = (t_out[k-1], t_out[k]), p = fip)
            sol = solve(prob, fip.diffeq.alg, reltol=1e-3)
            fip.geostate.dudt = sol(t_out[k], Val{1})
        else
            @inbounds for t in t_out[k-1]:fip.diffeq.dt:t_out[k]
                update_diagnostics!(fip.geostate.dudt, fip.geostate.u, fip, t)
                explicit_euler!(fip.geostate.u, fip.geostate.dudt, fip.diffeq.dt)
                write_out!(fip, k)
            end
        end
        write_out!(fip, k)
    end

    fip.out.computation_time += time()-t1
    return nothing
end

"""
    init(fip)

Initialize `ode::CoupledODEs`, aimed to be used in [`step!`](@ref).
"""
init(fip::FastIsoProblem) = CoupledODEs(update_diagnostics!,
    fip.geostate.u, fip; fip.diffeq)

"""
    init(fip)

Step `fip::FastIsoProblem` over `tspan` and based on `ode::CoupledODEs`, typically
obtained by [`init`](@ref).
"""
function step!(fip::FastIsoProblem{T, M}, ode::CoupledODEs,
    tspan::Tuple{T, T}) where {T<:AbstractFloat, M<:Matrix{T}}

    fip.out.computation_time -= time()
    dt = tspan[2] - tspan[1]
    X, t = trajectory(ode, dt, fip.geostate.u; t0 = tspan[1], Δt = dt)
    fip.geostate.u .= reshape(X[2, :], fip.Omega.Nx, fip.Omega.Ny)
    update_diagnostics!(fip.geostate.dudt, fip.geostate.u, fip, t[2])
    fip.out.computation_time += time()
end

"""
    update_diagnostics!(dudt, u, fip, t)

Update all the diagnotisc variables, i.e. all fields of `fip.geostate` apart
from the displacement, which requires an integrator.
"""
function update_diagnostics!(dudt::M, u::M, fip::FastIsoProblem{T, M}, t::T,
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    # Order really matters here!

    # Make sure that integrated viscous displacement satisfies BC.
    fip.Omega.bc!(u, fip.Omega.Nx, fip.Omega.Ny)
    
    if fip.internal_loadupdate
        update_loadcolumns!(fip, fip.tools.Hice(t))
    end

    # Only update the geoid and sea level if geostate is interactive.
    # As integration requires smaller time steps than diagnostics,
    # only update geostate every fip.geostate.dt
    if (t / fip.geostate.dt >= fip.geostate.countupdates)
        # if elastic update placed after geoid, worse match with (Spada et al. 2011)
        update_elasticresponse!(fip)
        if fip.interactive_sealevel
            update_geoid!(fip)
            update_sealevel!(fip)
        end
        fip.geostate.countupdates += 1
    end

    update_bedrock!(fip, u)
    dudt_isostasy!(dudt, u, fip, t)
    return nothing
end

"""
    dudt_isostasy!()

Update the displacement rate `dudt` of the viscous response.
"""
function dudt_isostasy!(
    dudt::AbstractMatrix{T},
    u::AbstractMatrix{T},
    fip::FastIsoProblem{T, M},
    t::T,
) where {T<:AbstractFloat, M<:AbstractMatrix{T}}

    Omega = fip.Omega
    rhs = - fip.c.g .* columnanom_full(fip)

    if fip.tools.negligible_gradD
        biharmonic_u = Omega.biharmonic .* (fip.tools.pfft * u)
        rhs += - fip.p.litho_rigidity .* real.( fip.tools.pifft * biharmonic_u )
    else
        dudxx, dudyy = Omega.fdxx(u), Omega.fdyy(u)
        Mxx = -fip.p.litho_rigidity .* (dudxx + fip.p.litho_poissonratio .* dudyy)
        Myy = -fip.p.litho_rigidity .* (dudyy + fip.p.litho_poissonratio .* dudxx)
        Mxy = -fip.p.litho_rigidity .* (1 - fip.p.litho_poissonratio) .* Omega.fdxy(u)
        rhs += Omega.fdxx(Mxx) + Omega.fdyy(Myy) + 2 .* Omega.fdxy(Mxy)
    end

    # dudt[:, :] .= real.(fip.tools.pifft * ((fip.tools.pfft * rhs) ./
    #     Omega.pseudodiff)) ./ (2 .* fip.p.effective_viscosity)
    dudt[:, :] .= real.(fip.tools.pifft * ((fip.tools.pfft * (rhs ./ 
        (2 .* fip.p.effective_viscosity)) ) ./ Omega.pseudodiff))
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
        u .-= ( view(u,1,1) + view(u,Nx,1) + view(u,1,Ny) + view(u,Nx,Ny) ) / 4
    end
    return u
end

"""
    edge_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at borders of domain is 0.
Same as [^Bueler2007].
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
    update_elasticresponse!(fip::FastIsoProblem)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from (Farell 1972), see [FastIsoTools](@ref).
"""
function update_elasticresponse!(fip::FastIsoProblem{T, M}
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    rgh = correct_surfacedisctortion(columnanom_load(fip), fip)
    fip.geostate.ue .= samesize_conv(fip.tools.elasticgreen, rgh, fip.Omega, no_bc) #edge_bc
    return nothing
end