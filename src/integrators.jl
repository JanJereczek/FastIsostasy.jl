using LinearAlgebra

abstract type AbstractIntegrator end

mutable struct Tsitouras54{
    A,  # Array
    T,  # Float
} <: AbstractIntegrator
    u0::A
    u1::A
    u2::A
    k1::A
    k2::A
    k3::A
    k4::A
    k5::A
    k6::A
    tol::T
    dt::T
    dt_min::T
    dt_max::T
end

function Tsitouras54(
    u0::Vector{T};
    tol = T(1e-6),
    dt_init = T(1e-3),
    dt_min = T(1e-6),
    dt_max = T(1.0),
) where T<:AbstractFloat
    return Tsitouras54(
        copy(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        tol,
        dt_init,
        dt_min,
        dt_max,
    )
end

mutable struct BogackiShampine32{A, T} <: AbstractIntegrator
    u0::A
    u1::A
    u2::A
    k1::A
    k2::A
    k3::A
    k4::A
    tol::T
    dt::T
    dt_min::T
    dt_max::T
end

function BogackiShampine32(
    u0::Vector{T};
    tol = T(1e-6),
    dt_init = T(1e-3),
    dt_min = T(1e-6),
    dt_max = T(1.0),
) where T<:AbstractFloat
    return BogackiShampine32(
        copy(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        similar(u0),
        tol,
        dt_init,
        dt_min,
        dt_max,
    )
end

mutable struct ODE{A, V, T, I}
    f::Function
    t::T
    u::A
    t_out::V
    u_out::Vector{A}
    i_out::Int
    integrator::I
end

function ODE(
    f,
    t_span[1],
    integrator;
    t_out = [],
)

    T = eltype(integrator.u0)
    u0 = integrator.u0

    if t_out == sort(t_out)
        u_out = [similar(u0) for _ in 1:length(t_out)]
    else
        error("t_out must be sorted!")
    end

    return ODE(
        f,
        T(t_span[1]),
        copy(u0),
        T.(t_out),
        u_out,
        1,
        integrator,
    )
end

function step!(integrator::Tsitouras54, f, t, u, dt)

    @. integrator.k1 = f(t, u)
    @. integrator.k2 = f(
        t + 0.161*dt,
        u + 0.161*dt*integrator.k1,
    )
    @. integrator.k3 = f(
        t + 0.327*dt,
        u + dt*0.327*(integrator.k1 + integrator.k2),
    )
    @. integrator.k4 = f(
        t + 0.5*dt,
        u + dt*0.5*(integrator.k1 + integrator.k3)
    )
    @. integrator.k5 = f(
        t + 0.805*dt,
        u + dt*(0.805*integrator.k1 - 1.607*integrator.k3 + 1.602*integrator.k4)
    )
    @. integrator.k6 = f(
        t + dt,
        u + dt*(0.106*integrator.k1 - 0.061*(integrator.k3 + integrator.k4) + 0.894*integrator.k5)
    )

    @. integrator.u2 = u + dt * 
        (7*(integrator.k1 + integrator.k6) +
        32*(integrator.k3 + integrator.k5) +
        12*integrator.k4)/90
    @. integrator.u1 = u + dt * 
        (32*(integrator.k3 + integrator.k5 - integrator.k6) +
        12*integrator.k4)/90
end

function step!(integrator::BogackiShampine32, f, t, u, dt)

    @. integrator.k1 = f(t, u)
    @. integrator.k2 = f(
        t + 0.5*dt,
        u + 0.5*dt*integrator.k1,
    )
    @. integrator.k3 = f(
        t + 0.75*dt,
        u + 0.75*dt*integrator.k2,
    )
    @. integrator.k4 = f(
        t + dt,
        u + dt*(2*integrator.k1 + 3*integrator.k2 + 4*integrator.k3)/9,
    )

    @. integrator.u1 = u + dt *
        (7/24*integrator.k1 + 1/4*integrator.k2 + 1/3*integrator.k3 + 1/8*integrator.k4)
    @. integrator.u2 = u + dt * (2*integrator.k1 + 3*integrator.k2 + 4*integrator.k3)/9
end

"""
Integrate ODE with adaptive time stepping.
"""
function step!(ode, t_end)

    while ode.t < t_end
        while ode.t < ode.t_out[ode.i_out]
            ode.integrator.dt = min(ode.integrator.dt, t_end - ode.t, ode.t_out[ode.i_out] - ode.t)
            
            step!(ode.integrator, ode.f, ode.t, ode.u, ode.integrator.dt)
            
            # Error estimation and adaptive stepping
            err = norm(ode.integrator.u2 .- ode.integrator.u1)
            if err < ode.integrator.tol
                @. ode.u = ode.integrator.u2
                ode.t += ode.integrator.dt
                ode.integrator.dt = min(ode.integrator.dt_max, ode.integrator.dt *
                    (ode.integrator.tol / err)^0.2)
            else
                ode.integrator.dt *= 0.5
            end
        end

        # Store output if t_out is reached
        copyto!(ode.u_out[ode.i_out], ode.u)
        ode.i_out += 1
        @show ode.t ode.integrator.dt
    end
end

f1(t, x) = -0.5 .* x
bs32 = BogackiShampine32([1.0], tol = 1e-3, dt_init = 0.1, dt_min = 1e-6, dt_max = 0.5)
ode32 = ODE(f1, 0.0, bs32, t_out = 0:1:5)
@b step!(ode32, 5.0)
u32 = [u[1] for u in ode32.u_out]
u_analytic = [exp(-0.5*t) for t in ode32.t_out]
check32 = isapprox(u32, u_analytic, atol = 1e-3) # should be true

tsit54 = Tsitouras54([1.0], tol = 1e-3, dt_init = 0.1, dt_min = 1e-6, dt_max = 0.5)
ode54 = ODE(f1, 0.0, tsit54, t_out = 0:0.1:5)
@b step!(ode54, 5.0)
u54 = [u[1] for u in ode54.u_out]
check54 = isapprox(u54, u_analytic, atol = 1e-3) # should be true
