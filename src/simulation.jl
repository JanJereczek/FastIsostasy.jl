#########################################################
# Options
#########################################################
"""
$(TYPEDSIGNATURES)

Control the options of `integrator<:AbstractIntegrator`.

# Fields
 - `alg`: the [`AbstractIntegrator`](@ref) used to integrate the ODE forward in time,
   one of `BS3Integrator()` (adaptive, default), `Tsit5Integrator()` (adaptive) or `EulerIntegrator()`
   (fixed step, requires `dt_min`).
 - `reltol`: the relative error tolerance of the adaptive controller.
 - `abstol`: the absolute error tolerance of the adaptive controller.
 - `dt_min`: fixed step size for `EulerIntegrator`, and a lower bound on the adaptive
   step size otherwise.
 - `dt0`: initial step size for the adaptive backend (`nothing` picks a
   conservative default).
"""
@kwdef struct DiffEqOptions{S}
    alg::S = BS3Integrator()
    reltol::AbstractFloat = 1.0f-5
    abstol::AbstractFloat = 1.0f-6
    dt_min::Union{Real,Nothing} = nothing
    dt0::Union{Real,Nothing} = nothing
end

"""
$(TYPEDSIGNATURES)

Control options relative to solving a [`Simulation`](@ref).

# Fields
 - `diffeq`: the [`DiffEqOptions`](@ref) controlling the ODE solver.
 - `dt_sparse_diagnostics`: the time interval between updates of the diagnostics variables (elastic displacement, sea-surface elevation, etc.).
 - `verbose`: whether to print information about the simulation progress.
 - `transition`: the [`AbstractTransition`](@ref) used to smooth the transition between grounded and floating ice, and between ocean and land.
"""
@kwdef struct SolverOptions{TR<:AbstractTransition}
    diffeq::DiffEqOptions = DiffEqOptions()
    dt_sparse_diagnostics::Float64 = 10.0
    verbose::Bool = true
    transition::TR = SharpTransition()
end

"""
$(TYPEDSIGNATURES)

Control the timing of the simulation and store the time evolution of the computation time.

# Fields
 - `t`: the current simulation time.
 - `t_span`: the time span of the simulation.
 - `t_vec`: the vector of times at which the computation time was recorded.
 - `t_computation_0`: the time at which the computation started.
 - `t_computation`: the vector of computation times corresponding to `t_vec`.
"""
mutable struct Timer{T}
    t::T
    t_span::Tuple{T,T}
    t_vec::Vector{T}
    t_computation_0::T
    t_computation::Vector{T}
end

function Timer(t_span; T = Float32)
    return Timer(T(t_span[1]), T.(t_span), T[], T(0), T[])
end

function t_computation!(tt::Timer)
    t_elapsed = time() - tt.t_computation_0
    if tt.t > tt.t_span[1]
        push!(tt.t_computation, t_elapsed)
        push!(tt.t_vec, tt.t)
    end
    return nothing
end

#########################################################
# Problem definition
#########################################################

"""
$(TYPEDSIGNATURES)

A superstruct needed for the forward integration of the model.

# Fields
 - `domain`: the [`AbstractDomain`](@ref) defining the spatial discretization.
 - `c`: the [`PhysicalConstants`](@ref) defining the physical constants of the model.
 - `bcs`: the [`BoundaryConditions`](@ref) defining the boundary conditions of the model.
 - `sealevel`: the [`RegionalSeaLevel`](@ref) defining the sea level evolution.
 - `solidearth`: the [`SolidEarth`](@ref) defining the solid earth properties.
 - `opts`: the [`SolverOptions`](@ref) controlling the solver options.
 - `tools`: the [`GIATools`](@ref) providing tools for GIA computations.
 - `ref`: the [`ReferenceState`](@ref) defining the reference state of the model.
 - `now`: the [`CurrentState`](@ref) defining the current state of the model.
 - `ncout`: the [`NetcdfOutput`](@ref) controlling the NetCDF output.
 - `nout`: the [`NativeOutput`](@ref) controlling the native output.
 - `timer`: the [`Timer`](@ref) controlling and recording timing information.
 - `simobs`: a vector of [`SimulatedObservable`](@ref) defining simulated observables to be computed during integration.
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
    TM,     # <:Timer
    VO,     # <:AbstractVector{<:SimulatedObservable} (inverse/observables.jl)
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
    timer::TM
    simobs::VO
end

function Simulation(
    domain,         # RegionalDomain
    bcs,            # BoundaryConditions
    sealevel,       # RegionalSeaLevel
    solidearth,     # SolidEarth
    t_span;          # Time span
    T = eltype(domain.R),
    opts = SolverOptions(),
    u_ref = zeros(domain),
    ue_ref = zeros(domain),
    dz_ss_ref = zeros(domain),
    z_b_ref = fill(1.0f6, domain),
    ncout = NetcdfOutput(domain, T[], ""),
    nout = NativeOutput(t = T[]),
    c = PhysicalConstants{T}(),
    simobs = SimulatedObservable[],
)

    if (sealevel.load isa NoSealevelLoad)
        nothing
    elseif (sum(solidearth.maskactive) > 0.6 * domain.nx * domain.ny)
        error(
            "Mask defining regions of active load must not cover more than 60%" *
            " of the cells when using an interactive sea level.",
        )
    end

    tools = GIATools(domain, c, solidearth)
    timer = Timer(t_span, T = T)

    # Initialise the reference state
    H_ice_ref = kernelzeros(domain)
    apply_bc!(H_ice_ref, timer.t, bcs.ice_thickness)

    u_ref, ue_ref, dz_ss_ref, z_b_ref, H_ice_ref = kernelpromote(
        [u_ref, ue_ref, dz_ss_ref, z_b_ref, H_ice_ref],
        domain.arraykernel,
    )
    z_ss_ref = sealevel.bsl.ref.z .+ dz_ss_ref

    tr = opts.transition
    if domain.use_cuda
        maskgrounded = get_maskgrounded(H_ice_ref, z_b_ref, z_ss_ref, c, tr)
        maskocean = get_maskocean(z_ss_ref, z_b_ref, maskgrounded, tr)
    else
        maskgrounded = collect(get_maskgrounded(H_ice_ref, z_b_ref, z_ss_ref, c, tr))
        maskocean = collect(get_maskocean(z_ss_ref, z_b_ref, maskgrounded, tr))
    end

    H_af_ref = height_above_floatation(H_ice_ref, z_b_ref, z_ss_ref, c, tr)
    H_water_ref = watercolumn(H_ice_ref, maskgrounded, z_b_ref, z_ss_ref, c, tr)
    ref = ReferenceState(
        u_ref,
        ue_ref,
        H_ice_ref,
        H_af_ref,
        H_water_ref,
        z_b_ref,
        z_ss_ref,
        T(0),
        T(0),
        T(0),
        maskgrounded,
        maskocean,
    )
    now = CurrentState(domain, ref, sealevel.bsl.z)

    return Simulation(
        domain,
        c,
        bcs,
        sealevel,
        solidearth,
        opts,
        tools,
        ref,
        now,
        ncout,
        deepcopy(nout),
        timer,
        simobs,
    )
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
        "native t_out" => sim.nout.t,
        "nc t_out" => sim.ncout.t,
        "n simulated observables" => length(sim.simobs),
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
# Output writing
#####################################################

"""
$(TYPEDSIGNATURES)

A function to be called by the integrator at each time step to write the output to NetCDF files.
"""
function nc_affect!(integrator)
    sim = integrator.p

    if occursin(".nc", sim.ncout.filename)
        sim.opts.verbose && println(
            "Saving nc output at index $(sim.ncout.k), sim year $(integrator.t)...",
        )

        if (:u_x in sim.ncout.vars3D) || (:u_y in sim.ncout.vars3D)
            thinplate_horizontal_displacement!(
                sim.now.u_x,
                sim.now.u_y,
                sim.now.u + sim.now.ue,
                sim.solidearth.litho_thickness,
                sim.domain,
            )
        end

        write_nc!(sim)
        sim.ncout.k += 1
    end
end

"""
$(TYPEDSIGNATURES)

A function to be called by the integrator at each time step to write the output to native files.
"""
function nout_affect!(integrator)
    sim = integrator.p
    sim.opts.verbose &&
        println("Saving native output at simulation year $(integrator.t)...")

    if (:u_x in sim.nout.vars) || (:u_y in sim.nout.vars)
        thinplate_horizontal_displacement!(
            sim.now.u_x,
            sim.now.u_y,
            sim.now.u + sim.now.ue,
            sim.solidearth.litho_thickness,
            sim.domain,
        )
    end

    write_out!(sim.nout, sim.now)
    sim.nout.k += 1
end

#####################################################
# Forward integration
#####################################################

"""
$(TYPEDSIGNATURES)

Initialize the simulation problem by computing the diagnostics variables.
"""
function init_problem!(sim::Simulation)
    update_V_af!(sim, sim.sealevel.volume_contribution)
    update_V_den!(sim, sim.sealevel.density_contribution)
    update_V_pov!(sim, sim.sealevel.adjustment_contribution)
    total_volume(sim)
    update_diagnostics!(sim.now.dudt, sim.now.u, sim, sim.timer.t)
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

    sim.timer.t = t

    # CAUTION: Order really matters here!
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
    update_diagnostics = (
        ((t - sim.timer.t_span[1]) / sim.opts.dt_sparse_diagnostics) >=
        sim.now.count_sparse_updates
    )   # +1

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

    t_computation!(sim.timer)  # Add computation time
    return nothing
end