#########################################################
# Options
#########################################################

@kwdef struct DiffEqOptions
    alg::Tsit5 = Tsit5()    # For now we limit the options
    reltol::Float64 = 1e-3
end

"""
    Options

Return a struct containing the options relative to solving a [`FastIsoProblem`](@ref).
"""
@kwdef struct SolverOptions
    deformation_model::Symbol = :lv_elva  # :lv_elva! or :lv_elra! or :lv_elra
    interactive_sealevel::Bool = false
    internal_loadupdate::Bool = true
    internal_bsl_update::Bool = true
    diffeq::DiffEqOptions = DiffEqOptions()
    dt_diagnostics::Float64 = 10.0
    verbose::Bool = false
end

#########################################################
# Problem definition
#########################################################

"""
    FastIsoProblem(Omega, c, p, t_out)
    FastIsoProblem(Omega, c, p, t_out, Hice)
    FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice)

Return a struct containing all the other structs needed for the forward integration of the
model over `Omega::ComputationDomain` with parameters `c::PhysicalConstants` and
`p::LayeredEarth`. The outputs are stored at `t_out::Vector{<:AbstractFloat}`.
"""
struct FastIsoProblem{
    T<:AbstractFloat,
    L<:Matrix{T},
    M<:KernelMatrix{T},
    B<:BoolMatrix,
    C<:ComplexMatrix{T},
    FP<:ForwardPlan{T},
    IP<:InversePlan{T},
    O<:Output}

    Omega::ComputationDomain{T, L, M}
    c::PhysicalConstants{T}
    p::LayeredEarth{T, M}
    opts::SolverOptions
    tools::FastIsoTools{T, M, C, FP, IP}
    ref::ReferenceState{T, M, B}
    now::CurrentState{T, M, B}
    ncout::NetcdfOutput{Float32}
    out::O
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real};
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Creating some placeholders in case of an external update of the load.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [null(Omega), null(Omega)]
    return FastIsoProblem(Omega, c, p, t_out, t_Hice_snapshots, Hice_snapshots; kwargs...)
end

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    Hice::KernelMatrix{T};
    kwargs...,
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
    # Constant interpolator in case viscosity is fixed over time.
    t_Hice_snapshots = [extrema(t_out)...]
    Hice_snapshots = [Hice, Hice]
    return FastIsoProblem(Omega, c, p, t_out, t_Hice_snapshots, Hice_snapshots; kwargs...)
end

zero_bsl(T, t) = linear_interpolation(T.([extrema(t)...]), T.([0.0, 0.0]),
    extrapolation_bc = Flat())

function FastIsoProblem(
    Omega::ComputationDomain{T, L, M},
    c::PhysicalConstants{T},
    p::LayeredEarth{T, M},
    t_out::Vector{<:Real},
    t_Hice_snapshots::Vector{T},
    Hice_snapshots::Vector{<:KernelMatrix{T}};
    opts::SolverOptions = SolverOptions(),
    u_0::KernelMatrix{T} = null(Omega),
    ue_0::KernelMatrix{T} = null(Omega),
    z_ss_0::KernelMatrix{T} = null(Omega),
    b_0::KernelMatrix{T} = null(Omega),
    bsl_itp = zero_bsl(T, t_out),
    maskactive::BoolMatrix = kernelcollect(Omega.K .< Inf, Omega),
    output_file::String = "",
    output::String = "nothing",
) where {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}

    if !isa(opts.diffeq.alg, Tsit5) && !isa(opts.diffeq.alg, SimpleEuler)
        error("Provided algorithm for solving ODE is not supported.")
    end

    if opts.interactive_sealevel & (sum(maskactive) > 0.6 * Omega.Nx * Omega.Ny)
        error("Mask defining regions of active load must not cover more than 60%"*
            " of the cells when using an interactive sea level.")
    end

    tools = FastIsoTools(Omega, c, p, t_Hice_snapshots, Hice_snapshots, bsl_itp)

    # Initialise the reference state
    H_ice_0 = kernelnull(Omega)
    piecewise_linear_interpolate!(H_ice_0, t_out[1], tools.Hice)
    # H_ice_0 = tools.Hice(t_out[1])
    bsl_0 = tools.bsl(t_out[1])
    u_0, ue_0, z_ss_0, b_0, H_ice_0 = kernelpromote([u_0, ue_0,
        z_ss_0, b_0, H_ice_0], Omega.arraykernel)

    if Omega.use_cuda
        maskgrounded = get_maskgrounded(H_ice_0, b_0, z_ss_0, c)
        maskocean = get_maskocean(z_ss_0, b_0, maskgrounded)
    else
        maskgrounded = collect(get_maskgrounded(H_ice_0, b_0, z_ss_0, c))
        maskocean = collect(get_maskocean(z_ss_0, b_0, maskgrounded))
    end

    H_af_0 = height_above_floatation(H_ice_0, b_0, z_ss_0, c)
    H_water_0 = watercolumn(H_ice_0, maskgrounded, b_0, z_ss_0, c)
    ref = ReferenceState(u_0, ue_0, H_ice_0, H_af_0, H_water_0, b_0, bsl_0, z_ss_0,
        T(0.0), T(0.0), T(0.0), maskgrounded, maskocean, Omega.arraykernel(maskactive))
    now = CurrentState(Omega, ref)
    ncout = NetcdfOutput(Omega, t_out, output_file)

    if output == "sparse"
        out = SparseOutput(Omega, t_out)
    elseif output == "intermediate"
        out = IntermediateOutput(Omega, t_out)
    else
        out = MinimalOutput(t_out, T[])
    end
    return FastIsoProblem(Omega, c, p, opts, tools, ref, now, ncout, out)
end


function Base.show(io::IO, ::MIME"text/plain", fip::FastIsoProblem)
    Omega, p = fip.Omega, fip.p
    println(io, "FastIsoProblem")
    descriptors = [
        "Wx, Wy" => [Omega.Wx, Omega.Wy],
        "dx, dy" => [Omega.dx, Omega.dy],
        "extrema(effective viscosity)" => extrema(p.effective_viscosity),
        "extrema(lithospheric thickness)" => extrema(p.litho_thickness),
    ]
    padlen = maximum(length(d[1]) for d in descriptors) + 2
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end
end

#####################################################
# Forward integration
#####################################################
"""
    solve!(fip)

Solve the isostatic adjustment problem defined in `fip::FastIsoProblem`.
"""
function solve!(fip::FastIsoProblem{T, L, M, B, C, FP, IP}) where {
    T<:AbstractFloat,
    L<:Matrix{T},
    M<:KernelMatrix{T},
    B<:BoolMatrix,
    C<:ComplexMatrix{T},
    FP<:ForwardPlan{T},
    IP<:InversePlan{T}}

    if !(fip.opts.internal_loadupdate)
        error("`solve!` does not support external updating of the load. Use `step!` instead.")
    end

    if fip.opts.deformation_model == :lv_elra
        throw(ArgumentError("LV-ELRA is not implemented yet."))
    end
    
    t1 = time()
    update_diagnostics!(fip.now.dudt, fip.now.u, fip, fip.out.t[1])
    (length(fip.ncout.filename) > 3) && write_nc!(fip.ncout, fip.now, fip.now.k)
    prob = ODEProblem(update_diagnostics!, fip.now.u, extrema(fip.out.t), fip)
    nc_callback = DiscreteCallback(nc_condition, nc_affect!)
    solve(prob, fip.opts.diffeq.alg, reltol=fip.opts.diffeq.reltol, saveat=fip.out.t,
        tstops=fip.out.t, callback=nc_callback)
    fip.ncout.computation_time += time()-t1
    return nothing
end

function init_problem(fip::FastIsoProblem)
    if !(fip.opts.internal_loadupdate)
        error("`solve!` does not support external updating of the load. Use `step!` instead.")
    end

    if fip.opts.deformation_model == :lv_elra
        throw(ArgumentError("LV-ELRA is not implemented yet."))
    end
    
    t1 = time()
    update_diagnostics!(fip.now.dudt, fip.now.u, fip, fip.out.t[1])
    (length(fip.ncout.filename) > 3) && write_nc!(fip.ncout, fip.now, fip.now.k)
    prob = ODEProblem(update_diagnostics!, fip.now.u, extrema(fip.out.t), fip)
    nc_callback = DiscreteCallback(nc_condition, nc_affect!)
    return prob, nc_callback, t1
end

nc_condition(_, t, integrator) = t in integrator.p.out.t

function nc_affect!(integrator)
    fip = integrator.p
    println("Saving at $(integrator.t) years...")
    fip.now.k += 1

    if fip.Omega.use_cuda == false
        thinplate_horizontal_displacement!(fip.now.u_x, fip.now.u_y,
            fip.now.u + fip.now.ue, fip.p.litho_thickness, fip.Omega)
    end
    
    if length(integrator.p.ncout.filename) > 3
        write_nc!(integrator.p.ncout, integrator.p.now, integrator.p.now.k)
    end

    if !(integrator.p.out isa MinimalOutput)
        write_out!(integrator.p.out, integrator.p.now, integrator.p.now.k)
    end
end

"""
    init_integrator(fip)
"""
function init_integrator(fip::FastIsoProblem)
    prob, _, _ = init_problem(fip)
    integrator = init(prob, fip.opts.diffeq.alg, reltol=fip.opts.diffeq.reltol,
        saveat=fip.out.t)
    return integrator
end