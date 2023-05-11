

function kalman_invert(kalmaninv::KalmanInversion)

    # Generating noisy observations
    y_t = zeros(length(kalmaninv.y), kalmaninv.n_samples)
    μ = zeros(length(kalmaninv.y))
    covH = reshape( kalmaninv.scale_obscov ./                # 10000.0
        (Hcylinder[forced_idx] .+ 1), sum(forced_idx) )
    Γy = convert(Array, Diagonal(vcat([covH for j in eachindex(results.viscous)[2:end]]...)) )
    for i in 1:n_samples
        y_t[:, i] = U .+ rand(MvNormal(μ, Γy))
    end
    truth = Observations.Observation(y_t, Γy, ["Wiens"])
    truth_sample = truth.mean

    priors = combine_distributions([constrained_gaussian( "p_$(i)",
        kalmaninv.params_mean_prior, kalmaninv.params_var_prior,
        kalmaninv.params_lowerbound, kalmaninv.params_upperbound) for i in 1:nparams])

    # Here we also could use process = Inversion()
    process = Unscented(mean(priors), cov(priors);
        α_reg = kalmaninv.α_reg, update_freq = kalmaninv.update_freq)

    ukiobj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)

    err = zeros(kalmaninv.N_iter)
    for n in 1:N_iter
        ϕ_n = get_ϕ_final(priors, ukiobj)       # Params in physical/constrained space
        println("size: ", size(ϕ_n), ",  mean parameter value: $(mean(ϕ_n))")
        G_n = [fastiso(ϕ_n[:, i]) for i in 1:size(ϕ_n)[2]]      # Evaluate forward map
        G_ens = hcat(G_n...)
        EnsembleKalmanProcesses.update_ensemble!(ukiobj, G_ens)
        err[n] = get_error(ukiobj)[end]
        println("Iteration: $n, Error: $(err[n]), norm(Cov): $(norm(ukiobj.process.uu_cov[n]))")
    end
    return priors, ukiobj
end

# ϕ_n = get_ϕ_final(priors, ukiobj)


# For now, method is fixed to UKI
struct KalmanInversion
    y::Vector{Matrix}
    t::Vector
    Omega::ComputationDomain
    c::PhysicalConstants
    p::MultilayerEarth
    Hice::Vector{Matrix}
    nparams::Int
    params_mean_prior::Real
    params_var_prior::Real
    params_lowerbound::Real
    params_upperbound::Real
    N_iter::Int
    α_reg::Real
    update_freq::Int
    n_samples::Int
    obsnoise_mean
    obsnoise_var
end


function KalmanInversion(y::Matrix, paramsconfig::NamedTuple)
    T = Float64
    L = T(3000e3)
    Omega = ComputationDomain(L, n)
    c = PhysicalConstants()
    p = MultilayerEarth()

    R = T(2000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hice = uniform_ice_cylinder(Omega, R, H)
    Htol = 1.0

    kalmaninv = KalmanInversion(y, Omega, c, p, Hice)

    return kalman(kalmaninv)
end


"""
    find_invertibles()

Find points of parameter field that can be inverted. We here assume that 
"""
function find_invertibles()
end

function forcingproportional_covariance()
end

# Actually, this should not be diagonal because there is a correlation between points.
function uncorrelated_obs_covariance()
end

function correlated_obs_covariance()
end

#####################################################
# Optimization
#####################################################
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
