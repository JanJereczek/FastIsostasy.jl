"""
    explicit_euler!()

Update the state `u` by performing an explicit Euler integration of its derivative `dudt`
over a time step `dt`.
"""
function explicit_euler!(u::M, dudt::M, dt::T,
    ) where {T<:AbstractFloat, M<:AbstractMatrix{T}}
    u .+= dudt .* dt
    return nothing
end

function explicit_rk4!(fi::FastIso{T, M}, f::Function, dt::T, t::T) where
    {T<:AbstractFloat, M<:AbstractMatrix{T}}
    k1 = dt .* f(fi.geostate.dudt, fi.geostate.u, fi, t)
    k2 = dt .* f(fi.geostate.dudt, fi.geostate.u .+ k1/2, fi, t + dt/2)
    k3 = dt .* f(fi.geostate.dudt, fi.geostate.u .+ k2/2, fi, t + dt/2)
    k4 = dt .* f(fi.geostate.dudt, fi.geostate.u .+ k2, fi, t + dt)
    fi.geostate.u .+= (k1 + 2*k2 + 2*k3 + k4)/6
    return nothing
end