#####################################################
# Forward integration
#####################################################
"""
    solve!(fip)

Solve the isostatic adjustment problem defined in `fip::FastIsoProblem`.
"""
function solve!(fip::FastIsoProblem)

    if !(fip.opts.internal_loadupdate)
        error("`solve!` does not support external updating of the load. Use `step!` instead.")
    end
    t1 = time()
    t_out = fip.out.t

    # Make a first diagnotisc update to store these values. (k=1)
    update_diagnostics!(fip.now.dudt, fip.now.u, fip, t_out[1])
    write_out!(fip, 1)

    # Initialize dummy ODEProblem and perform integration.
    dummy = ODEProblem(update_diagnostics!, fip.now.u, (0.0, 1.0), fip)
    @inbounds for k in eachindex(t_out)[2:end]
        
        fip.now.k = k - 1
        if fip.opts.verbose
            println("Computing until t = $(Int(round(seconds2years(t_out[k])))) years...")
        end

        if fip.opts.diffeq.alg != SimpleEuler()
            prob = remake(dummy, u0 = fip.now.u, tspan = (t_out[k-1], t_out[k]), p = fip)
            sol = solve(prob, fip.opts.diffeq.alg, reltol=fip.opts.diffeq.reltol)
            fip.now.dudt = sol(t_out[k], Val{1})
        else
            @inbounds for t in t_out[k-1]:fip.opts.diffeq.dt:t_out[k]
                update_diagnostics!(fip.now.dudt, fip.now.u, fip, t)
                simple_euler!(fip.now.u, fip.now.dudt, fip.opts.diffeq.dt)
            end
        end
        write_out!(fip, k)
        fip.now.countupdates = 0    # reset to update sl at beginning of next solve()

    end

    fip.out.computation_time += time()-t1
    return nothing
end

"""
    init(fip)

Initialize an `ode::CoupledODEs`, aimed to be used in [`step!`](@ref).
"""
init(fip::FastIsoProblem) = CoupledODEs(update_diagnostics!, fip.now.u, fip; fip.opts.diffeq)

"""
    step!(fip)

Step `fip::FastIsoProblem` over `tspan` and based on `ode::CoupledODEs`, typically
obtained by [`init`](@ref).
"""
function step!(fip::FastIsoProblem{T, M}, ode::CoupledODEs,
    tspan::Tuple{T, T}) where {T<:AbstractFloat, M<:Matrix{T}}

    fip.out.computation_time -= time()
    dt = tspan[2] - tspan[1]
    X, t = trajectory(ode, dt, fip.now.u; t0 = tspan[1], Δt = dt)
    fip.now.u .= reshape(X[2, :], fip.Omega.Nx, fip.Omega.Ny)
    update_diagnostics!(fip.now.dudt, fip.now.u, fip, t[2])
    fip.out.computation_time += time()
    return nothing
end

"""
    update_diagnostics!(dudt, u, fip, t)

Update all the diagnotisc variables, i.e. all fields of `fip.now` apart
from the displacement, which requires an integrator.
"""
function update_diagnostics!(dudt::M, u::M, fip::FastIsoProblem{T, L, M, C, FP, IP}, t::T,
    ) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}

    # Order really matters here!

    # Make sure that integrated viscous displacement satisfies BC.
    # corner_bc!(u, fip.Omega.Nx, fip.Omega.Ny, fip.now.ucorner)
    apply_bc!(u, fip.Omega.bc_matrix, fip.Omega.nbc)

    # Update load columns if interpolator available
    if fip.opts.internal_loadupdate
        update_loadcolumns!(fip, fip.tools.Hice(t))
    end

    # Regardless of update method for column, update the anomalies!
    columnanom_load!(fip)

    # Only update the geoid and sea level if now is interactive.
    # As integration requires smaller time steps than diagnostics,
    # only update geostate every fip.now.dt
    if (((t - fip.out.t[fip.now.k]) / fip.opts.dt_sl) >= fip.now.countupdates) ||
        t ≈ fip.out.t[fip.now.k + 1]
        # if elastic update placed after geoid, worse match with (Spada et al. 2011)
        update_elasticresponse!(fip)
        columnanom_litho!(fip)
        if fip.opts.interactive_sealevel
            if fip.opts.internal_bsl_update
                update_bsl!(fip)
            else
                fip.now.bsl = fip.tools.bsl(seconds2years(t))
            end
            update_geoid!(fip)
            update_seasurfaceheight!(fip)
            update_maskocean!(fip)
        end
        fip.now.countupdates += 1
    end
    columnanom_full!(fip)
    dudt_isostasy!(dudt, u, fip, t)
    columnanom_mantle!(fip)
    update_bedrock!(fip, u)

    return nothing
end

function update_ucorner!(fip::FastIsoProblem)
    fip.now.ucorner = -fip.now.bsl * fip.c.rho_seawater / fip.c.rho_uppermantle
end

"""
    dudt_isostasy!()

Update the displacement rate `dudt` of the viscous response.
"""
function dudt_isostasy!(dudt::M, u::M, fip::FastIsoProblem{T, L, M, C, FP, IP}, t::T) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}

    Omega, P = fip.Omega, fip.tools.prealloc
    P.rhs .= -fip.c.g .* fip.now.columnanoms.full
    update_second_derivatives!(P.uxx, P.uyy, P.ux, P.uxy, u, Omega)
    P.Mxx .= - fip.p.litho_rigidity .* (P.uxx + fip.p.litho_poissonratio .* P.uyy)
    P.Myy .= - fip.p.litho_rigidity .* (P.uyy + fip.p.litho_poissonratio .* P.uxx)
    P.Mxy .= - fip.p.litho_rigidity .* (1 - fip.p.litho_poissonratio) .* P.uxy
    update_second_derivatives!(P.Mxxxx, P.Myyyy, P.Mxyx, P.Mxyxy, P.Mxx, P.Myy,
        P.Mxy, Omega)
    P.rhs += P.Mxxxx + P.Myyyy + 2 .* P.Mxyxy

    P.fftrhs .= complex.(P.rhs .* Omega.K ./ (2 .* fip.p.effective_viscosity))
    fip.tools.pfft! * P.fftrhs
    P.ifftrhs .= P.fftrhs ./ Omega.pseudodiff
    fip.tools.pifft! * P.ifftrhs
    dudt .= real.(P.ifftrhs)
    
    # corner_bc!(dudt, fip.Omega.Nx, fip.Omega.Ny, 0.0)
    apply_bc!(dudt, fip.Omega.bc_matrix, fip.Omega.nbc)

    return nothing
end

#####################################################
# BCs
#####################################################

"""
    apply_bc!(u::M, bcm::M, nbc::T)
A generic function to update `u` such that the boundary conditions are respected
on average, which is the only way to impose them for Fourier collocation.
"""
function apply_bc!(u::M, bcm::M, nbc::T) where {T <: AbstractFloat, M<:KernelMatrix{T}}
    u .-= sum(u .* bcm) / nbc
end

"""
    corner_bc!(u, Nx, Ny, offset)

Shift `u` s.t. its mean value at the corner points is `offset`.
"""
function corner_bc!(u::KernelMatrix{T}, Nx::Int, Ny::Int, offset::T) where {T<:AbstractFloat}
    allowscalar() do
        u .-= ( u[1, 1] + u[Nx, 1] + u[1, Ny] + u[Nx, Ny] ) / 4
    end
    u .+= offset
    return u
end

"""
    edge_bc!(u, Nx, Ny)

Shift `u` s.t. its mean value at on the edge is 0. Same as in [bueler-fast-2007](@cite).
"""
function edge_bc!(u::KernelMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    allowscalar() do
        u .-= sum( view(u, 1, :) + view(u, Nx, :) + view(u, :, Ny) + view(u, :, 1) ) /
            (2*Nx + 2*Ny)
    end
    return u
end

function no_mean_bc!(u::KernelMatrix{<:AbstractFloat}, Nx::Int, Ny::Int)
    u .= u .- mean(u)
    return u
end

#####################################################
# Elastic response
#####################################################
"""
    update_elasticresponse!(fip::FastIsoProblem)

Update the elastic response by convoluting the Green's function with the load anom.
To use coefficients differing from [^Farrell1972], see [FastIsoTools](@ref).
"""
function update_elasticresponse!(fip::FastIsoProblem{T, L, M, C, FP, IP}) where
    {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}, C<:ComplexMatrix{T},
    FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    fip.now.ue .= samesize_conv(fip.now.columnanoms.load .* fip.Omega.K .^ 2,
        fip.tools.elasticconvo, fip.Omega)
    return nothing
end

#####################################################
# Mechanics utils
#####################################################

function compute_shearmodulus(m::ReferenceEarthModel)
    return m.density .* (m.Vsv + m.Vsh) ./ 2
end

function maxwelltime_scaling(layer_viscosities, layer_shearmoduli)
    return layer_shearmoduli[end] ./ layer_shearmoduli .* layer_viscosities
end

function maxwelltime_scaling!(layer_viscosities, layer_boundaries, m::ReferenceEarthModel)
    mu = compute_shearmodulus(m)
    layer_meandepths = (layer_boundaries[:, :, 1:end-1] + layer_boundaries[:, :, 2:end]) ./ 2
    layer_meandepths = cat(layer_meandepths, layer_boundaries[:, :, end], dims = 3)
    mu_itp = linear_interpolation(m.depth, mu)
    layer_meanshearmoduli = layer_viscosities ./ 1e21 .* mu_itp.(layer_meandepths)
    layer_viscosities .*= layer_meanshearmoduli[:, :, end] ./ layer_meanshearmoduli
end


"""
    get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}

Compute rigidity `D` based on thickness `t`, Young modulus `E` and Poisson ration `nu`.
"""
function get_rigidity(t::T, E::T, nu::T) where {T<:AbstractFloat}
    return (E * t^3) / (12 * (1 - nu^2))
end

"""
    get_effective_viscosity(
        layer_viscosities::Vector{KernelMatrix{T}},
        layers_thickness::Vector{T},
        Omega::ComputationDomain{T, M},
    ) where {T<:AbstractFloat}

Compute equivalent viscosity for multilayer model by recursively applying
the formula for a halfspace and a channel from Lingle and Clark (1975).
"""
function get_effective_viscosity(
    Omega::ComputationDomain{T, M},
    layer_viscosities::Array{T, 3},
    layers_thickness::Array{T, 3},
    mantle_poissonratio::T,
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    incompressible_poissonratio = T(0.5)
    compressibility_scaling = (1 + incompressible_poissonratio) / (1 + mantle_poissonratio)

    # Recursion has to start with half space = n-th layer:
    effective_viscosity = layer_viscosities[:, :, end]
    if size(layer_viscosities, 3) > 1
        @inbounds for i in axes(layer_viscosities, 3)[1:end-1]
            channel_viscosity = layer_viscosities[:, :, end - i]
            channel_thickness = layers_thickness[:, :, end - i + 1]
            viscosity_ratio = channel_viscosity ./ effective_viscosity
            viscosity_scaling = three_layer_scaling(
                Omega,
                viscosity_ratio,
                channel_thickness,
            )
            effective_viscosity .*= viscosity_scaling
        end
    end
    effective_compressible_viscosity = effective_viscosity .* compressibility_scaling

    correct_shearmoduluschange = true

    if correct_shearmoduluschange
        corrected_viscosity = seakon_calibration(effective_compressible_viscosity)
    else
        corrected_viscosity = effective_compressible_viscosity
    end
    return corrected_viscosity
end

function seakon_calibration(eta::Matrix{T}) where {T<:AbstractFloat}
    return exp.(log10.(T(1e21) ./ eta)) .* eta
end

"""
    three_layer_scaling(Omega::ComputationDomain, kappa::T, visc_ratio::T,
        channel_thickness::T)

Return the viscosity scaling for a three-layer model and based on a the wave
number `kappa`, the `visc_ratio` and the `channel_thickness`.
Reference: Bueler et al. 2007, below equation 15.
"""
function three_layer_scaling(
    Omega::ComputationDomain{T, M},
    visc_ratio::Matrix{T},
    channel_thickness::Matrix{T},
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    # kappa is the wavenumber of the harmonic load. (see Cathles 1975, p.43)
    # we assume this is related to the size of the domain!
    kappa = T(π) / mean([Omega.Wx, Omega.Wy])

    C = cosh.(channel_thickness .* kappa)
    S = sinh.(channel_thickness .* kappa)

    num1 = 2 .* visc_ratio .* C .* S
    num2 = (1 .- visc_ratio .^ 2) .* channel_thickness .^ 2 .* kappa .^ 2
    num3 = visc_ratio .^ 2 .* S .^ 2 + C .^ 2

    denum1 = (visc_ratio .+ 1 ./ visc_ratio) .* C .* S
    denum2 = (visc_ratio .- 1 ./ visc_ratio) .* channel_thickness .* kappa
    denum3 = S .^ 2 + C .^ 2
    
    return (num1 + num2 + num3) ./ (denum1 + denum2 + denum3)
end


"""
    build_greenintegrand(distance::Vector{T}, 
        greenintegrand_coeffs::Vector{T}) where {T<:AbstractFloat}

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function build_greenintegrand(
    distance::Vector{T},
    greenintegrand_coeffs::Vector{T},
) where {T<:AbstractFloat}

    greenintegrand_interp = linear_interpolation(distance, greenintegrand_coeffs)
    compute_greenintegrand_entry_r(r::T) = get_loadgreen(
        r, distance, greenintegrand_coeffs, greenintegrand_interp)
    greenintegrand_function(x::T, y::T) = compute_greenintegrand_entry_r( get_r(x, y) )
    return greenintegrand_function
end

"""
    get_loadgreen(r::T, rm::Vector{T}, greenintegrand_coeffs::Vector{T},     
        interp_greenintegrand_::Interpolations.Extrapolation) where {T<:AbstractFloat}

Compute the integrands of the Green's function resulting from a load at a given
`distance` and based on provided `greenintegrand_coeffs`.
Reference: Deformation of the Earth by surface Loads, Farell 1972, table A3.
"""
function get_loadgreen(
    r::T,
    rm::Vector{T},
    greenintegrand_coeffs::Vector{T},
    interp_greenintegrand_::Interpolations.Extrapolation,
) where {T<:AbstractFloat}

    if r < 0.01
        return greenintegrand_coeffs[1] / ( rm[2] * T(1e12) )
    elseif r > rm[end]
        return T(0.0)
    else
        return interp_greenintegrand_(r) / ( r * T(1e12) )
    end
end

"""
    get_elasticgreen(Omega, quad_support, quad_coeffs)

Integrate load response over field by using 2D quadrature with specified
support points and associated coefficients.
"""
function get_elasticgreen(
    Omega::ComputationDomain{T, M},
    greenintegrand_function::Function,
    quad_support::Vector{T},
    quad_coeffs::Vector{T},
) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    dx, dy = Omega.dx, Omega.dy
    elasticgreen = fill(T(0), Omega.Nx, Omega.Ny)

    @inbounds for i = 1:Omega.Nx, j = 1:Omega.Ny
        p = i - Omega.Mx - 1
        q = j - Omega.My - 1
        elasticgreen[j, i] = quadrature2D(
            greenintegrand_function,
            quad_support,
            quad_coeffs,
            p*dx,
            (p+1)*dx,
            q*dy,
            (q+1)*dy,
        )
    end
    return Omega.arraykernel(elasticgreen)
end