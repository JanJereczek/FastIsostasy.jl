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
    k1 = dt .* f(fip.geostate.dudt, fip.geostate.u, fip, t)
    k2 = dt .* f(fip.geostate.dudt, fip.geostate.u .+ k1/2, fip, t + dt/2)
    k3 = dt .* f(fip.geostate.dudt, fip.geostate.u .+ k2/2, fip, t + dt/2)
    k4 = dt .* f(fip.geostate.dudt, fip.geostate.u .+ k2, fip, t + dt)
    fip.geostate.u .+= (k1 + 2*k2 + 2*k3 + k4)/6
    return nothing
end