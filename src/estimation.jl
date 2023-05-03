using Statistics, Optim

struct ViscOptim
    t_out::Vector
    sstruct::SuperStruct
    U::Vector{Matrix}
    dUdt::Vector{Matrix}
end

# Here we denote X as the estimated viscosity field and
# U the field of observable resulting from a ground truth viscosity field Y.
function optimize_viscosity(Omega, Hice, t_out, U)
    vo = init_optim(Omega, Hice, t_out, U)
    x0 = fill(1e21, Omega.N * Omega.N)
    res = optimize(vo, x0, LBFGS())
    return Matrix(reshape(minimizer(res), Omega.N, Omega.N))
end

function init_optim(Omega, Hice, t_out, U; activegeostate = false)
    c = PhysicalConstants()
    p = MultilayerEarth(Omega, c)
    sstruct = init_superstruct(Omega, c, p, t_out, Hice)

    if var(diff(t_out)) < 1e-3  # yr
        dt = mean(diff(t_out))
    else
        error("The provided time vector is not evenly spaced!")
    end
    dUdt = (U[3:end] - U[1:end-2]) ./ (2 * dt)

    return ViscOptim(years2seconds.(t_out[2:end-1]), sstruct, U[2:end-1], dUdt)
end

function (vo::ViscOptim)(x)
    vo.sstruct.p.effective_viscosity .= Matrix(reshape(x, Omega.N, Omega.N))
    return integrated_rmse(vo)
end

function integrated_rmse(vo::ViscOptim)
    dudt = zeros(eltype(vo.dUdt[1]), size(vo.dUdt[1]))
    e = 0.0
    for i in eachindex(vo.dUdt)
        dudt_isostasy!(dudt, vo.U[i], vo.sstruct, vo.t_out[i])
        e += rmse(dudt, dUdt[i+1])
    end
    return e
end

function rmse(UX, UY)
    return sum( (UX - UY).^2 )
end