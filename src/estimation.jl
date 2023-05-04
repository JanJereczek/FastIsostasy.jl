"""

    ViscOptim()

A `struct` that contains all the variables related to opitmization.
If applied as a function, it returns the cost.
"""
struct ViscOptim
    t_out::Vector
    sstruct::SuperStruct
    U::Vector{Matrix}
    dUdt::Vector{Matrix}
end

# Here we denote X as the estimated viscosity field and
# U the field of observable resulting from a ground truth viscosity field Y.
function optimize_viscosity(
    Omega::ComputationDomain,
    Hice,
    t_out,
    U,
    opts;
    x0 = 21.0, # fill(21.0, Omega.N * Omega.N)
    # x0 = fill(20.0, Omega.N * Omega.N) + rand(Omega.N * Omega.N)
)
    vo = init_optim(Omega, Hice, t_out, U)

    # lx = fill(19.0, Omega.N * Omega.N)
    # ux = fill(21.0, Omega.N * Omega.N)
    # dfc = TwiceDifferentiableConstraints(lx, ux)

    res = optimize(vo, [x0], LBFGS(), opts)
    return res
end

function extract_minimizer()
    return Matrix(reshape(res.minimizer, Omega.N, Omega.N))
end

function init_optim(Omega, Hice, t_out, U; active_geostate = false)
    c = PhysicalConstants()
    p = MultilayerEarth(Omega, c)
    eta = [p.effective_viscosity for t in t_out]
    sstruct = init_superstruct(Omega, c, p, t_out, Hice, t_out, eta, active_geostate)

    if var(diff(t_out)) < 1e-3  # yr
        dt = mean(diff(t_out))
    else
        error("The provided time vector is not evenly spaced!")
    end
    dUdt = (U[3:end] - U[1:end-2]) ./ (2 * dt)

    return ViscOptim(years2seconds.(t_out[2:end-1]), sstruct, U[2:end-1], dUdt)
end

function (vo::ViscOptim)(x)
    # vo.sstruct.p.effective_viscosity .= 10.0 .^ Matrix(reshape(x, vo.sstruct.Omega.N, vo.sstruct.Omega.N))
    vo.sstruct.p.effective_viscosity .= 10.0 .^ fill(x[1], vo.sstruct.Omega.N, vo.sstruct.Omega.N)
    return integrated_rmse(vo)
end

function integrated_rmse(vo::ViscOptim)
    dudt = zeros(eltype(vo.dUdt[1]), size(vo.dUdt[1]))
    e = 0.0 # TODO introduce a regularization term
    for i in eachindex(vo.dUdt)
        dudt_isostasy!(dudt, vo.U[i], vo.sstruct, vo.t_out[i])
        e += rmse(
            m_per_sec2mm_per_yr.(dudt),
            m_per_sec2mm_per_yr.(vo.dUdt[i]),
        ) / (1e5 * length(vo.t_out))
    end
    display(e)
    display(mean(log10.(vo.sstruct.p.effective_viscosity[1])))
    return e
end

function rmse(UX, UY)
    return sum( (UX - UY).^2 )
end