#########################################################
# Options
#########################################################
@kwdef struct DiffEqOptions{S<:ODEsolvers}
    alg::S = BS3()
    reltol::AbstractFloat = 1f-3
end

"""
    Options

Return a struct containing the options relative to solving a [`FastIsoProblem`](@ref).
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
    FastIsoProblem(Omega, c, p, t_out)
    FastIsoProblem(Omega, c, p, t_out, Hice)
    FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice)

Return a struct containing all the other structs needed for the forward integration of the
model over `Omega::RegionalComputationDomain` with parameters `c::PhysicalConstants` and
`p::SolidEarthParameters`. The outputs are stored at `t_out::Vector{<:AbstractFloat}`.
"""
struct FastIsoProblem{
    CD,     # <:AbstractComputationDomain
    PC,     # <:PhysicalConstants
    BCS,    # <:ProblemBCs
    EM,     # <:SolidEarthModel
    LE,     # <:SolidEarthParameters
    SO,     # <:SolverOptions
    FIT,    # <:FastIsoTools
    RS,     # <:ReferenceState
    CS,     # <:CurrentState
    NCO,    # <:NetcdfOutput
    NO,     # <:NativeOutput
}
    Omega::CD
    c::PC
    bcs::BCS
    em::EM
    p::LE
    opts::SO
    tools::FIT
    ref::RS
    now::CS
    ncout::NCO
    nout::NO
end

function FastIsoProblem(
    Omega,  # RegionalComputationDomain
    em,     # SolidEarthModel
    p;      # SolidEarthParameters
    T = eltype(Omega.R),
    bcs = ProblemBCs(Omega),
    opts = SolverOptions(),
    u_0 = null(Omega),
    ue_0 = null(Omega),
    z_ss_0 = null(Omega),
    z_b_0 = null(Omega),
    maskactive = kernelcollect(Omega.K .< Inf, Omega),
    ncout = NetcdfOutput(Omega, T[], ""),
    nout = NativeOutput(t = T[]),
    c = PhysicalConstants{T}(),
    bsl = ConstantBSL(),
)

    if (bcs.ocean_load isa NoOceanLoad)
        nothing
    elseif (sum(maskactive) > 0.6 * Omega.nx * Omega.ny)
        error("Mask defining regions of active load must not cover more than 60%"*
            " of the cells when using an interactive sea level.")
    end

    tools = FastIsoTools(Omega, c, p)

    # Initialise the reference state
    H_ice_0 = kernelnull(Omega)
    apply_bc!(H_ice_0, nout.t[1], bcs.ice_thickness)

    u_0, ue_0, z_ss_0, z_b_0, H_ice_0 = kernelpromote([u_0, ue_0,
        z_ss_0, z_b_0, H_ice_0], Omega.arraykernel)

    if Omega.use_cuda
        maskgrounded = get_maskgrounded(H_ice_0, z_b_0, z_ss_0, c)
        maskocean = get_maskocean(z_ss_0, z_b_0, maskgrounded)
    else
        maskgrounded = collect(get_maskgrounded(H_ice_0, z_b_0, z_ss_0, c))
        maskocean = collect(get_maskocean(z_ss_0, z_b_0, maskgrounded))
    end

    H_af_0 = height_above_floatation(H_ice_0, z_b_0, z_ss_0, c)
    H_water_0 = watercolumn(H_ice_0, maskgrounded, z_b_0, z_ss_0, c)
    ref = ReferenceState(u_0, ue_0, H_ice_0, H_af_0, H_water_0, z_b_0, z_ss_0,
        T(0), T(0), T(0), maskgrounded, maskocean, Omega.arraykernel(maskactive))
    now = CurrentState(Omega, ref, bsl)

    return FastIsoProblem(Omega, c, bcs, em, p, opts, tools, ref, now, ncout, nout)
end

function Base.show(io::IO, ::MIME"text/plain", fip::FastIsoProblem)
    Omega, p = fip.Omega, fip.p
    descriptors = [
        "Computation domain" => typeof(Omega),
        "Physical constants" => typeof(fip.c),
        "Problem BCs" => typeof(fip.bcs),
        "Earth model" => typeof(fip.em),
        "Layered Earth" => typeof(p),
        "Solver options" => typeof(fip.opts),
        "FastIsoTools" => typeof(fip.tools),
        "Reference state" => typeof(fip.ref),
        "Current state" => typeof(fip.now),
        "Netcdf output" => typeof(fip.ncout),
        "Native output" => typeof(fip.nout),
        "t_out" => fip.nout.t,
        "nx, ny" => [Omega.nx, Omega.ny],
        "dx, dy" => [Omega.dx, Omega.dy],
        "Wx, Wy" => [Omega.Wx, Omega.Wy],
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

nc_condition(_, t, integrator) = (t in integrator.p.ncout.t) && 
    (integrator.p.now.k < length(integrator.p.ncout.t))

nout_condition(_, t, integrator) = (t in integrator.p.nout.t)

function nc_affect!(integrator)
    fip = integrator.p

    if occursin(".nc", fip.ncout.filename)
        println("Saving at nc output at index $(fip.now.k), simulation year $(integrator.t)...")
        fip.now.k += 1

        if (:u_x in fip.ncout.vars3D) || (:u_y in fip.ncout.vars3D)
            thinplate_horizontal_displacement!(fip.now.u_x, fip.now.u_y,
                fip.now.u + fip.now.ue, fip.p.litho_thickness, fip.Omega)
        end

        write_nc!(fip)
    end
end

function nout_affect!(integrator)
    fip = integrator.p
    println("Saving native output at simulation year $(integrator.t)...")
    # @show mean(vcat(fip.now.u[1, :], fip.now.u[end, :], fip.now.u[:, 1], fip.now.u[:, end]))

    if (:u_x in fip.nout.vars) || (:u_y in fip.nout.vars)
        thinplate_horizontal_displacement!(fip.now.u_x, fip.now.u_y,
            fip.now.u + fip.now.ue, fip.p.litho_thickness, fip.Omega)
    end

    write_out!(fip.nout, fip.now)
end

function write_nc!(fip::FastIsoProblem)
    if occursin(".nc", fip.ncout.filename) > 3
        write_nc!(fip.ncout, fip.now, fip.now.k)
    end
end

#####################################################
# Forward integration
#####################################################

"""
    solve!(fip)

Solve the isostatic adjustment problem defined in `fip::FastIsoProblem`.
"""
function solve!(fip::FastIsoProblem)
    init_problem!(fip)
    t1 = time()
    prob = ODEProblem(update_diagnostics!, fip.now.u, extrema(fip.nout.t), fip)
    ncout_callback = DiscreteCallback(nc_condition, nc_affect!)
    nout_callback = DiscreteCallback(nout_condition, nout_affect!)
    out_callback = CallbackSet(ncout_callback, nout_callback)
    solve(prob, fip.opts.diffeq.alg, reltol=fip.opts.diffeq.reltol,
        saveat=fip.nout.t[end:end], tstops=fip.nout.t, callback=out_callback,
        progress = fip.opts.verbose)
    fip.nout.computation_time = time()-t1
    return nothing
end


"""
    init_integrator(fip)
"""
function init_integrator(fip::FastIsoProblem)
    init_problem!(fip)
    prob = ODEProblem(update_diagnostics!, fip.now.u, extrema(fip.nout.t), fip)
    ncout_callback = DiscreteCallback(nc_condition, nc_affect!)
    nout_callback = DiscreteCallback(nout_condition, nout_affect!)
    out_callback = CallbackSet(ncout_callback, nout_callback)
    integrator = init(prob, fip.opts.diffeq.alg, reltol=fip.opts.diffeq.reltol,
        saveat=fip.nout.t[end:end], tstops=fip.nout.t, callback=out_callback)
    return integrator
end

function init_problem!(fip::FastIsoProblem)
    update_diagnostics!(fip.now.dudt, fip.now.u, fip, fip.nout.t[1])
    write_nc!(fip)
    return nothing
end