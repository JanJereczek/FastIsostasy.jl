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
end

"""
$(TYPEDSIGNATURES)

Return a struct containing the options relative to solving a [`Simulation`](@ref).
"""
@kwdef struct SolverOptions
    diffeq::DiffEqOptions = DiffEqOptions()
    dt_diagnostics::Float64 = 10.0
    verbose::Bool = true
end

#########################################################
# Problem definition
#########################################################

"""
$(TYPEDSIGNATURES)

    Simulation(domain, c, p, t_out)
    Simulation(domain, c, p, t_out, Hice)
    Simulation(domain, c, p, t_out, t_Hice, Hice)

Return a struct containing all the other structs needed for the forward integration of the
model over `domain::RegionalDomain` with parameters `c::PhysicalConstants` and
`p::SolidEarthParameters`. The outputs are stored at `t_out::Vector{<:AbstractFloat}`.
"""
struct Simulation{
    CD,     # <:AbstractDomain
    PC,     # <:PhysicalConstants
    BCS,    # <:BoundaryConditions
    GM,     # <:Model
    SEP,    # <:SolidEarthParameters
    SO,     # <:SolverOptions
    FIT,    # <:GIATools
    RS,     # <:ReferenceState
    CS,     # <:CurrentState
    NCO,    # <:NetcdfOutput
    NO,     # <:NativeOutput
    TS,     # <:Tuple{<:Real, <:Real}
}
    domain::CD
    c::PC
    bcs::BCS
    model::GM
    p::SEP
    opts::SO
    tools::FIT
    ref::RS
    now::CS
    ncout::NCO
    nout::NO
    tspan::TS
end

function Simulation(
    domain,     # RegionalDomain
    model,      # Model
    p,          # SolidEarthParameters
    tspan;      # Time span
    T = eltype(domain.R),
    bcs = BoundaryConditions(domain),
    opts = SolverOptions(),
    u_ref = null(domain),
    ue_ref = null(domain),
    dz_ss_ref = null(domain),
    z_b_ref = null(domain),
    maskactive = kernelcollect(domain.K .< Inf, domain),
    ncout = NetcdfOutput(domain, T[], ""),
    nout = NativeOutput(t = T[]),
    c = PhysicalConstants{T}(),
)

    if (model.ocean_load isa NoOceanLoad)
        nothing
    elseif (sum(maskactive) > 0.6 * domain.nx * domain.ny)
        error("Mask defining regions of active load must not cover more than 60%"*
            " of the cells when using an interactive sea level.")
    end

    tools = GIATools(domain, c, p)

    # Initialise the reference state
    H_ice_ref = kernelnull(domain)
    apply_bc!(H_ice_ref, tspan[1], bcs.ice_thickness)

    u_ref, ue_ref, dz_ss_ref, z_b_ref, H_ice_ref = kernelpromote([u_ref, ue_ref,
        dz_ss_ref, z_b_ref, H_ice_ref], domain.arraykernel)
    z_ss_ref = model.bsl.ref.z .+ dz_ss_ref

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
        T(0), T(0), T(0), maskgrounded, maskocean, domain.arraykernel(maskactive))
    now = CurrentState(domain, ref, model.bsl.z)

    return Simulation(domain, c, bcs, model, p, opts, tools, ref, now, ncout, nout, tspan)
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    domain, p = sim.domain, sim.p
    descriptors = [
        "Computation domain" => typeof(domain),
        "Physical constants" => typeof(sim.c),
        "Problem BCs" => typeof(sim.bcs),
        "Earth model" => typeof(sim.model),
        "Layered Earth" => typeof(p),
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
        "extrema(effective viscosity)" => extrema(p.effective_viscosity),
        "extrema(lithospheric thickness)" => extrema(p.litho_thickness),
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
    (t >= integrator.p.ncout.t[integrator.p.ncout.k]) &&
    (integrator.p.ncout.k <= length(integrator.p.ncout.t))

nout_condition(_, t, integrator) = (t >= integrator.p.nout.t[integrator.p.nout.k])

function nc_affect!(integrator)
    sim = integrator.p

    if occursin(".nc", sim.ncout.filename)
        println("Saving at nc output at index $(sim.now.k), simulation year $(integrator.t)...")
        sim.ncout.k += 1

        if (:u_x in sim.ncout.vars3D) || (:u_y in sim.ncout.vars3D)
            thinplate_horizontal_displacement!(sim.now.u_x, sim.now.u_y,
                sim.now.u + sim.now.ue, sim.p.litho_thickness, sim.domain)
        end

        write_nc!(sim)
    end
end

function nout_affect!(integrator)
    sim = integrator.p
    println("Saving native output at simulation year $(integrator.t)...")
    # @show mean(vcat(sim.now.u[1, :], sim.now.u[end, :], sim.now.u[:, 1], sim.now.u[:, end]))
    sim.nout.k += 1

    if (:u_x in sim.nout.vars) || (:u_y in sim.nout.vars)
        thinplate_horizontal_displacement!(sim.now.u_x, sim.now.u_y,
            sim.now.u + sim.now.ue, sim.p.litho_thickness, sim.domain)
    end

    write_out!(sim.nout, sim.now)
end

#####################################################
# Forward integration
#####################################################

"""
$(TYPEDSIGNATURES)

Solve the isostatic adjustment problem defined in `sim::Simulation`.
"""
function run!(sim::Simulation)
    init_problem!(sim)
    t1 = time()
    prob = ODEProblem(update_diagnostics!, sim.now.u, sim.tspan, sim)
    ncout_callback = DiscreteCallback(nc_condition, nc_affect!)
    nout_callback = DiscreteCallback(nout_condition, nout_affect!)
    out_callback = CallbackSet(ncout_callback, nout_callback)
    solve(prob, sim.opts.diffeq.alg, reltol=sim.opts.diffeq.reltol,
        saveat=sim.nout.t[end:end], tstops=vcat(sim.nout.t, sim.ncout.t),
        callback=out_callback, progress=sim.opts.verbose)
    sim.nout.computation_time = time()-t1
    return nothing
end


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
    update_diagnostics!(sim.now.dudt, sim.now.u, sim, sim.tspan[1])
    return nothing
end


function write_nc!(sim::Simulation)
    if occursin(".nc", sim.ncout.filename)
        write_nc!(sim.ncout, sim.now, sim.now.k)
    end
end

"""
    update_diagnostics!(dudt, u, sim, t)

Update all the diagnotisc variables, i.e. all fields of `sim.now` apart
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
    columnanom_ice!(sim)                        # Compute associated column anomaly

    # apply_bc!(sim.now.H_sed, t, sim.bcs.)
    # columnanom_sediment!(sim)
    
    # As integration requires smaller time steps than what we typically want
    # for the elastic displacement and the sea-surface elevation,
    # we only update them every sim.opts.dt_diagnostics
    update_diagnostics = ((t / sim.opts.dt_diagnostics) >=
        sim.now.count_sparse_updates + 1)

    # if elastic update placed after dz_ss, worse match with (Spada et al. 2011)
    if update_diagnostics

        # Update the elastic response and the resulting anomaly in lithospheric column
        update_elasticresponse!(sim, sim.model.lithosphere)
        columnanom_litho!(sim)

        # Update barystatic sea level
        internal_update_bsl!(sim, sim.model.update_bsl)

        # Update perturbation of sea surface elevation according to new anomalies
        columnanom_load!(sim)
        columnanom_full!(sim)
        update_dz_ss!(sim, sim.model.sea_surface)

        # Update sea surface based on perturbation and BSL
        update_z_ss!(sim)

        # Update hieght above floatation and the resulting masks
        update_Haf!(sim)
        update_maskgrounded!(sim)
        update_maskocean!(sim)

        # Update the anomaly of seawater column
        columnanom_water!(sim, sim.model.ocean_load)
        
        # Count the sparse update
        sim.now.count_sparse_updates += 1
    end

    # Include the newly updated seawater column in the full column anomaly
    columnanom_load!(sim)
    columnanom_full!(sim)

    # Update the derivative of the viscous displacement based on the new load
    update_dudt!(dudt, u, sim, t, sim.model)
    sim.now.dudt .= dudt
    return nothing
end
