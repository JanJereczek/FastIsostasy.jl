struct SimpleEuler end

"""
    simple_euler!()

Update the state `u` by performing an explicit Euler integration of its derivative `dudt`
over a time step `dt`.
"""
function simple_euler!(u::M, dudt::M, dt::T,
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}
    u .+= dudt .* dt
    return nothing
end

function explicit_rk4!(fip::FastIsoProblem{T, M}, f::Function, dt::T, t::T) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    k1 = dt .* f(fip.now.dudt, fip.now.u, fip, t)
    k2 = dt .* f(fip.now.dudt, fip.now.u .+ k1/2, fip, t + dt/2)
    k3 = dt .* f(fip.now.dudt, fip.now.u .+ k2/2, fip, t + dt/2)
    k4 = dt .* f(fip.now.dudt, fip.now.u .+ k2, fip, t + dt)
    fip.now.u .+= (k1 + 2*k2 + 2*k3 + k4)/6
    return nothing
end



mutable struct CrankNicolson{T<:AbstractFloat, M<:KernelMatrix{T}}
    Xprevious::M
    Fprevious::M
    Xguess::M
    Xpreviousguess::M
    convergence_diff::T
    tol::T
    alpha::T
    dt::T
end

function CrankNicolson(Omega::ComputationDomain{T, L, M}; tol = 1e-3, alpha = 0.5,
    dt = years2seconds(10)) where {T<:AbstractFloat, L, M}
    return CrankNicolson(null(Omega), null(Omega), null(Omega), null(Omega),
        T(Inf), T(tol), T(alpha), T(dt))
end

function (cn::CrankNicolson)(x)
    cn.convergence_diff = Inf
    cn.Xprevious = copy(x)
    cn.Xguess = copy(xprevious)
    cn.Xpreviousguess = copy(xprevious)
    cn.Fprevious = dudt_isostasy!(dudt, cn.Xprevious, fip, t)

    while cn.convergence_diff > cn.tol
        cn.Xguess = cn.alpha * cn.Xguess + (1 - alpha) * (cn.Xprevious + dt/2 *
            (Fprevious + dudt_isostasy!(dudt, Xguess, fip, t)))
        cn.convergence_diff = mean( abs.(cn.Xguess - cn.Xpreviousguess) )
    end

end