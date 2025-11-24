#########################################################
# Options
#########################################################
"""
$(TYPEDSIGNATURES)

Contains:
- `alg::ODEsolvers`: the algorithm to integrate the ODE forward in time.
- `reltol`: the relative error tolerance of the integrator.
"""
@kwdef struct DiffEqOptions{S<:ODEsolvers}
    alg::S = BS3()
    reltol::AbstractFloat = 1f-5
    dt_min::Union{Real, Nothing} = nothing
end

"""
$(TYPEDSIGNATURES)

Return a struct containing the options relative to solving a [`Simulation`](@ref).
"""
@kwdef struct SolverOptions
    diffeq::DiffEqOptions = DiffEqOptions()
    dt_sparse_diagnostics::Float64 = 10.0
    verbose::Bool = true
end

#########################################################
# Problem definition
#########################################################

"""
$(TYPEDSIGNATURES)

    Simulation(domain, c, solidearth, t_out)
    Simulation(domain, c, solidearth, t_out, Hice)
    Simulation(domain, c, solidearth, t_out, t_Hice, Hice)

Return a struct containing all the other structs needed for the forward integration of the
model over `domain::RegionalDomain` with parameters `c::PhysicalConstants` and
`solidearth::SolidEarth`. The outputs are stored at `t_out::Vector{<:AbstractFloat}`.
"""
struct Simulation{
    CD,     # <:AbstractDomain
    PC,     # <:PhysicalConstants
    BCS,    # <:BoundaryConditions
    SL,     # <:RegionalSeaLevel
    SE,     # <:SolidEarth
    SO,     # <:SolverOptions
    TL,    # <:GIATools
    RS,     # <:ReferenceState
    CS,     # <:CurrentState
    NCO,    # <:NetcdfOutput
    NO,     # <:NativeOutput
    TS,     # <:Tuple{<:Real, <:Real}
}
    domain::CD
    c::PC
    bcs::BCS
    sealevel::SL
    solidearth::SE
    opts::SO
    tools::TL
    ref::RS
    now::CS
    ncout::NCO
    nout::NO
    tspan::TS
end

function Simulation(
    domain,         # RegionalDomain
    bcs,            # BoundaryConditions
    sealevel,       # RegionalSeaLevel
    solidearth;     # SolidEarth
    T = eltype(domain.R),
    tspan = extrema(bcs.ice_thickness.t_vec),
    opts = SolverOptions(),
    u_ref = zeros(domain),
    ue_ref = zeros(domain),
    dz_ss_ref = zeros(domain),
    z_b_ref = fill(1f6, domain),
    ncout = NetcdfOutput(domain, T[], ""),
    nout = NativeOutput(t = T[]),
    c = PhysicalConstants{T}(),
)

    if (sealevel.load isa NoSealevelLoad)
        nothing
    elseif (sum(solidearth.maskactive) > 0.6 * domain.nx * domain.ny)
        error("Mask defining regions of active load must not cover more than 60%"*
            " of the cells when using an interactive sea level.")
    end

    tools = GIATools(domain, c, solidearth)

    # Initialise the reference state
    H_ice_ref = kernelzeros(domain)
    apply_bc!(H_ice_ref, tspan[1], bcs.ice_thickness)

    u_ref, ue_ref, dz_ss_ref, z_b_ref, H_ice_ref = kernelpromote([u_ref, ue_ref,
        dz_ss_ref, z_b_ref, H_ice_ref], domain.arraykernel)
    z_ss_ref = sealevel.bsl.ref.z .+ dz_ss_ref

    if domain.use_cuda
        maskgrounded = get_maskgrounded(H_ice_ref, z_b_ref, z_ss_ref, c)
        maskocean = get_maskocean(z_ss_ref, z_b_ref, maskgrounded)
    else
        maskgrounded = collect(get_maskgrounded(H_ice_ref, z_b_ref, z_ss_ref, c))
        maskocean = collect(get_maskocean(z_ss_ref, z_b_ref, maskgrounded))
    end

    H_af_ref = height_above_floatation(H_ice_ref, z_b_ref, z_ss_ref, c)
    H_water_ref = watercolumn(H_ice_ref, maskgrounded, z_b_ref, z_ss_ref, c)
    ref = ReferenceState(u_ref, ue_ref, H_ice_ref, H_af_ref, H_water_ref, z_b_ref, z_ss_ref,
        T(0), T(0), T(0), maskgrounded, maskocean)
    now = CurrentState(domain, ref, sealevel.bsl.z)

    return Simulation(domain, c, bcs, sealevel, solidearth, opts, tools, ref, now,
        ncout, deepcopy(nout), tspan)
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    domain, solidearth = sim.domain, sim.solidearth
    descriptors = [
        "Computation domain" => typeof(domain),
        "Physical constants" => typeof(sim.c),
        "Problem BCs" => typeof(sim.bcs),
        "Sea level" => typeof(sim.sealevel),
        "Solid Earth" => typeof(solidearth),
        "Solver options" => typeof(sim.opts),
        "GIATools" => typeof(sim.tools),
        "Reference state" => typeof(sim.ref),
        "Current state" => typeof(sim.now),
        "Netcdf output" => typeof(sim.ncout),
        "Native output" => typeof(sim.nout),
        "t_out" => sim.nout.t,
        "nx, ny" => [domain.nx, domain.ny],
        "dx, dy" => [domain.dx, domain.dy],
        "Wx, Wy" => [domain.Wx, domain.Wy],
        "extrema(effective viscosity)" => extrema(solidearth.effective_viscosity),
        "extrema(lithospheric thickness)" => extrema(solidearth.litho_thickness),
    ]
    padlen = maximum(length(d[1]) for d in descriptors) + 2
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end

#####################################################
# I/O Callbacks
#####################################################

nc_condition(_, t, integrator) = (length(integrator.p.ncout.t) >= 1) &&
    (integrator.p.ncout.k <= length(integrator.p.ncout.t)) &&
    (t >= integrator.p.ncout.t[integrator.p.ncout.k])

nout_condition(_, t, integrator) = (length(integrator.p.nout.t) >= 1) &&
    (integrator.p.nout.k <= length(integrator.p.nout.t)) &&
    (t >= integrator.p.nout.t[integrator.p.nout.k])

function nc_affect!(integrator)
    sim = integrator.p

    if occursin(".nc", sim.ncout.filename)
        sim.opts.verbose && println("Saving nc output at index $(sim.ncout.k), sim year $(integrator.t)...")

        if (:u_x in sim.ncout.vars3D) || (:u_y in sim.ncout.vars3D)
            thinplate_horizontal_displacement!(sim.now.u_x, sim.now.u_y,
                sim.now.u + sim.now.ue, sim.solidearth.litho_thickness, sim.domain)
        end

        write_nc!(sim)
        sim.ncout.k += 1
    end
end

function nout_affect!(integrator)
    sim = integrator.p
    sim.opts.verbose && println("Saving native output at simulation year $(integrator.t)...")

    if (:u_x in sim.nout.vars) || (:u_y in sim.nout.vars)
        thinplate_horizontal_displacement!(sim.now.u_x, sim.now.u_y,
            sim.now.u + sim.now.ue, sim.solidearth.litho_thickness, sim.domain)
    end

    write_out!(sim.nout, sim.now)
    sim.nout.k += 1
end

#####################################################
# Forward integration
#####################################################

"""
$(TYPEDSIGNATURES)

Solve the isostatic adjustment problem defined in `sim::Simulation`.
"""
function run!(sim::Simulation)
    t1 = time()
    init_problem!(sim)
    prob = ODEProblem(update_diagnostics!, sim.now.u, sim.tspan, sim)
    ncout_callback = DiscreteCallback(nc_condition, nc_affect!)
    nout_callback = DiscreteCallback(nout_condition, nout_affect!)
    out_callback = CallbackSet(ncout_callback, nout_callback)

    if sim.opts.diffeq.dt_min isa Real
        if sim.opts.diffeq.alg isa Euler
            solve(prob, sim.opts.diffeq.alg, reltol=sim.opts.diffeq.reltol,
                saveat=[sim.tspan[end]], tstops=vcat(sim.nout.t, sim.ncout.t),
                callback=out_callback, progress=sim.opts.verbose,
                dtmin = sim.opts.diffeq.dt_min, force_dtmin = true,
                dt = sim.opts.diffeq.dt_min)
        else
            error("The `dt_min` option is only compatible with the Euler algorithm.")
        end
    else
        solve(prob, sim.opts.diffeq.alg, reltol=sim.opts.diffeq.reltol,
            saveat=[sim.tspan[end]], tstops=vcat(sim.nout.t, sim.ncout.t),
            callback=out_callback, progress=sim.opts.verbose)
    end
    sim.nout.computation_time = time()-t1
    return nothing
end

# In the best case, we would like something like:
# restart!(sim, tspan)                # restarts the simulation over tspan
# restart!(sim, tspan, refine = 2)    # same but refines the mesh by a factor of 2

# function run!(sim::Simulation, tspan)
#     if maximum(tspan) > maximum(sim.bcs.ice_thickness.t) ||
#         sim.bcs.ice_thickness.flat_bc == false
        
#         error("tspan must be larger than the maximum time of the ice thickness BC")
#     end
#     sim.tspan = tspan
#     run!(sim)
#     return nothing
# end

# function run!(sim::Simulation, tspan; refine_factor = 2)
#     domain = RegionalDomain(sim.domain.Wx, sim.domain.Wy,
#         refine_factor * sim.domain.nx, refine_factor * sim.domain.ny)
    
# end



"""
$(TYPEDSIGNATURES)

Initialise the integrator of `sim::Simulation`, which can be subsequently
integrated forward in time by using `step!`.
"""
function init_integrator(sim::Simulation)
    init_problem!(sim)
    prob = ODEProblem(update_diagnostics!, sim.now.u, extrema(sim.nout.t), sim)
    ncout_callback = DiscreteCallback(nc_condition, nc_affect!)
    nout_callback = DiscreteCallback(nout_condition, nout_affect!)
    out_callback = CallbackSet(ncout_callback, nout_callback)
    integrator = init(prob, sim.opts.diffeq.alg, reltol=sim.opts.diffeq.reltol,
        saveat=sim.nout.t[end:end], tstops=sim.nout.t, callback=out_callback)
    return integrator
end

function init_problem!(sim::Simulation)
    update_V_af!(sim, sim.sealevel.volume_contribution)
    update_V_den!(sim, sim.sealevel.density_contribution)
    update_V_pov!(sim, sim.sealevel.adjustment_contribution)
    total_volume(sim)
    update_diagnostics!(sim.now.dudt, sim.now.u, sim, sim.tspan[1])
    return nothing
end


function write_nc!(sim::Simulation)
    write_nc!(sim.ncout, sim.now, sim.ncout.k)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update all the diagnostics variables, i.e. all fields of `sim.now` apart
from the displacement, which requires an integrator.
"""
function update_diagnostics!(dudt, u, sim::Simulation, t)

    # CAUTION: Order really matters here!
    push!(sim.nout.t_steps_ode, t)                          # Add time step to output

    # Update the mantle anomaly and the bedrock elevation
    apply_bc!(u, sim.bcs.viscous_displacement)              # Make sure that u satisfies BC
    update_bedrock!(sim, u)
    columnanom_mantle!(sim)

    apply_bc!(sim.now.H_ice, t, sim.bcs.ice_thickness)      # Apply ice thickness BC
    update_Haf!(sim)
    columnanom_ice!(sim)                        # Compute associated column anomaly

    # apply_bc!(sim.now.H_sed, t, sim.bcs.)
    # columnanom_sediment!(sim)
    
    # As integration requires smaller time steps than what we typically want
    # for the elastic displacement and the sea-surface elevation,
    # we only update them every sim.opts.dt_sparse_diagnostics
    update_diagnostics = ((t / sim.opts.dt_sparse_diagnostics) >=
        sim.now.count_sparse_updates)   # +1

    # if elastic update placed after dz_ss, worse match with (Spada et al. 2011)
    if update_diagnostics

        # Update the elastic response and the resulting anomaly in lithospheric column
        update_elasticresponse!(sim, sim.solidearth.lithosphere)
        columnanom_litho!(sim)

        # Update barystatic sea level
        internal_update_bsl!(sim, sim.sealevel.update_bsl)
        update_dz_ss!(sim, sim.sealevel.surface)
        update_z_ss!(sim)

        # Update hieght above floatation and the resulting masks
        update_Haf!(sim)
        update_maskgrounded!(sim)
        update_maskocean!(sim)

        # Update the anomaly of seawater column
        columnanom_water!(sim, sim.sealevel.load)
        columnanom_ice!(sim)

        # Count the sparse update
        sim.now.count_sparse_updates += 1
    end

    # Include the newly updated seawater column in the full column anomaly
    columnanom_load!(sim)
    columnanom_full!(sim)

    # Update the derivative of the viscous displacement based on the new load
    update_dudt!(dudt, u, sim, t, sim.solidearth)
    sim.now.dudt .= dudt
    return nothing
end