
#####################################################
# Precomputation
#####################################################

"""

    precompute_fastiso(dt, Omega, p, c)

Return a `struct` containing pre-computed tools to perform forward-stepping. Takes the
time step `dt`, the ComputationDomain `Omega`, the solid-Earth parameters `p` and 
physical constants `c` as input.
"""
@inline function precompute_fastiso(
    Omega::ComputationDomain{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T};
    quad_precision::Int = 4,
) where {T<:AbstractFloat}

    quad_support, quad_coeffs = get_quad_coeffs(T, quad_precision)
    loadresponse = get_integrated_loadresponse(Omega, quad_support, quad_coeffs)

    Dx = mixed_fdx(p.litho_rigidity, Omega.dx)
    Dy = mixed_fdy(p.litho_rigidity, Omega.dy)
    Dxx = mixed_fdxx(p.litho_rigidity, Omega.dx)
    Dyy = mixed_fdyy(p.litho_rigidity, Omega.dy)
    Dxy = mixed_fdy( mixed_fdx(p.litho_rigidity, Omega.dx), Omega.dy )
    rhog = p.mean_density .* c.g

    if Omega.use_cuda
        Xgpu = CuArray(Omega.X)
        p1 = CUDA.CUFFT.plan_fft(Xgpu)
        p2 = CUDA.CUFFT.plan_ifft(Xgpu)
        # Dx, Dy, Dxx, Dyy, Dxy, rhog = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy, rhog])
        Dx, Dy, Dxx, Dyy, Dxy = convert2CuArray([Dx, Dy, Dxx, Dxy, Dyy])
    else
        p1, p2 = plan_fft(Omega.X), plan_ifft(Omega.X)
    end

    return PrecomputedFastiso(
        loadresponse,
        fft(loadresponse),
        p1,
        p2,
        Dx,
        Dy,
        Dxx,
        Dyy,
        Dxy,
        rhog,
    )
end

struct PrecomputedFastiso{T<:AbstractFloat}
    loadresponse::AbstractMatrix{T}
    fourier_loadresponse::AbstractMatrix{Complex{T}}
    pfft::AbstractFFTs.Plan
    pifft::AbstractFFTs.ScaledPlan
    Dx::AbstractMatrix{T}
    Dy::AbstractMatrix{T}
    Dxx::AbstractMatrix{T}
    Dyy::AbstractMatrix{T}
    Dxy::AbstractMatrix{T}
    rhog::T
end

#####################################################
# Forward integration
#####################################################

"""

    forward_isostasy!(Omega, t_out, u3D_elastic, u3D_viscous, sigma_zz, tools, p, c)

Integrates isostasy model over provided time vector `t_out` for a given
`ComputationDomain` `Omega`, pre-allocated arrays `u3D_elastic` and `u3D_viscous`
containing the solution, `sigma_zz` a time-interpolator ofthe vertical load applied
upon the bedrock, `tools` some pre-computed terms to speed up the computation, `p` some
`MultilayerEarth` parameters and `c` the `PhysicalConstants` of the problem.
"""
@inline function forward_isostasy!(
    Omega::ComputationDomain{T},
    t_out::AbstractVector{T},       # the output time vector
    u3D_elastic::AbstractArray{T, 3},
    u3D_viscous::AbstractArray{T, 3},
    dudt3D_viscous::AbstractArray{T, 3},
    sigma_zz_snapshots::Tuple{Vector{T}, Vector{Matrix{T}}},
    tools::PrecomputedFastiso{T},
    p::MultilayerEarth{T},
    c::PhysicalConstants{T};
    dt = fill(T(years2seconds(1.0)), length(t_out)-1)::AbstractVector{T},
) where {T}

    if Omega.use_cuda
        sigma_zz_values = convert2CuArray(sigma_zz_snapshots[2])
    else
        sigma_zz_values = sigma_zz_snapshots[2]
    end
    sigma_zz = linear_interpolation(sigma_zz_snapshots[1], sigma_zz_values)

    for i in eachindex(t_out)[1:end-1]
        t = t_out[i]
        dt_out = t_out[i+1] - t_out[i]
        u3D_elastic[:, :, i+1] .= compute_elastic_response(Omega, tools, -Array(sigma_zz(t)) ./ c.g )

        u3D_viscous[:, :, i+1], dudt3D_viscous[:, :, i+1] = forwardstep_viscous_response(
            Omega,
            t,
            dt[i],
            dt_out,
            u3D_viscous[:, :, i],
            dudt3D_viscous[:, :, i],
            sigma_zz,
            tools,
            p,
        )

    end
end

"""

    forwardstep_viscous_response(Omega, dt, dt_out, u_viscous, sigma_zz, tools, p)

Perform GIA computation one step forward w.r.t. the time vector used for storing the
output. This implies performing a number of in-between steps (for numerical stability
and accuracy) that are given by the ratio between `dt_out` and `dt`.
"""
@inline function forwardstep_viscous_response(
    Omega::ComputationDomain,
    t::T,
    dt::T,
    dt_out::T,
    u_viscous::AbstractMatrix{T},
    dudt_viscous::AbstractMatrix{T},
    sigma_zz::Interpolations.Extrapolation,
    tools::PrecomputedFastiso,
    p::MultilayerEarth,
) where {T<:AbstractFloat}

    if Omega.use_cuda
        u_viscous, dudt_viscous = convert2CuArray([u_viscous, dudt_viscous])
    end

    t_internal = range(t, stop = t + dt_out, step = dt)
    u_viscous_next = copy(u_viscous)
    dudt_viscous_next = copy(dudt_viscous)
    for i in eachindex(t_internal)[1:end-1]
        viscous_response!(
            Omega,
            t_internal[i],
            dt,
            u_viscous_next,
            dudt_viscous_next,
            sigma_zz,
            tools,
            p,
        )
        apply_bc!(u_viscous_next)
        apply_bc!(dudt_viscous_next)
    end

    if Omega.use_cuda
        u_viscous_next, dudt_viscous_next = convert2Array(
            [u_viscous_next, dudt_viscous_next])
    end
    return u_viscous_next, dudt_viscous_next
end

#####################################################
# RHS
#####################################################

"""

    viscous_response!(Omega, dt, u_current, sigma_zz, tools, c, p)

Return viscous response based on explicit Euler time discretization and Fourier 
collocation. Valid for multilayer parameters that can vary over x, y.
"""
@inline function viscous_response!(
    Omega::ComputationDomain,
    t::T,
    dt::T,
    u::AbstractMatrix{T},
    dudt::AbstractMatrix{T},
    sigma_zz::Interpolations.Extrapolation,
    tools::PrecomputedFastiso,
    p::MultilayerEarth,
) where {T<:AbstractFloat}

    uf = tools.pfft * u
    harmonic_uf = real.(tools.pifft * ( Omega.harmonic_coeffs .* uf ))
    biharmonic_uf = real.( tools.pifft * ( Omega.biharmonic_coeffs .* uf ) )

    term1 = sigma_zz(t)
    term2 = - tools.rhog .* u
    # term3 = - p.litho_rigidity .* biharmonic_uf
    term4 = - T(2) .* tools.Dx .* mixed_fdx(harmonic_uf, Omega.dx)
    term5 = - T(2) .* tools.Dy .* mixed_fdy(harmonic_uf, Omega.dy)
    term6 = - real.(tools.pifft * (Omega.harmonic_coeffs .* (tools.pfft * (p.litho_rigidity .* harmonic_uf))))
    term7 = tools.Dxx .* mixed_fdyy(u, Omega.dy)
    term8 = - T(2) .* tools.Dxy .* mixed_fdy( mixed_fdx(u, Omega.dx), Omega.dy )
    term9 = tools.Dyy .* mixed_fdxx(u, Omega.dx)

    rhs = term1 + term2 + # term3 + 
            term4 + term5 + term6 + (T(1) - p.litho_poissonratio) .*
            (term7 + term8 + term9)

    dudtf = (tools.pfft * rhs) ./ Omega.pseudodiff_coeffs
    dudt .= real.(tools.pifft * dudtf) ./ (T(2) .* p.effective_viscosity)
    u .+= dt .* dudt

end

#####################################################
# BCs
#####################################################

"""

    apply_bc(u)

Apply boundary condition on Fourier collocation solution.
Assume that mean deformation at corners of domain is 0.
Whereas Bueler et al. (2007) take the edges for this computation, we take the corners
because they represent the far-field better.
"""
@inline function apply_bc(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    u_bc = copy(u)
    return apply_bc!(u_bc)
end

@inline function apply_bc!(u::AbstractMatrix{T}) where {T<:AbstractFloat}
    CUDA.allowscalar() do
        u .-= (u[1,1] + u[1,end] + u[end,1] + u[end,end]) / T(4)
    end
end

#####################################################
# Elastic response
#####################################################
"""

    compute_elastic_response(tools, load)

For a computation domain `Omega`, compute the elastic response of the solid Earth
by convoluting the `load` with the Green's function stored in the pre-computed `tools`
(elements obtained from Farell 1972).
"""
@inline function compute_elastic_response(
    Omega::ComputationDomain,
    tools::PrecomputedFastiso,
    load::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return conv(load, tools.loadresponse)[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end

#####################################################
# Geoid response
#####################################################
"""

    compute_geoid_response(Omega, tools, load)

Compute the geoid response of by convoluting the `geoid_green` with the
`regional_mass_change` for a computation domain `Omega`.
"""
@inline function compute_geoid_response(
    Omega::ComputationDomain{T},
    geoid_green::AbstractMatrix{T},
    regional_mass_change::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return conv(
        geoid_green,
        regional_mass_change,
    )[Omega.N2:end-Omega.N2, Omega.N2:end-Omega.N2]
end

@inline function get_regional_mass_change(
    c::PhysicalConstants{T},
    p::MultilayerEarth{T},
    hi::AbstractMatrix{T},
    hw::AbstractMatrix{T},
    b::AbstractMatrix{T},
    hi0::AbstractMatrix{T},
    hw0::AbstractMatrix{T},
    b0::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return c.ice_density .* (hi - hi0) + c.water_density .* (hw - hw0) +
    p.mean_density .* (b - b0)
end

@inline function get_geoid_green(
    c::PhysicalConstants{T},
    theta::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return c.r_equator ./ c.mE ./ ( 2 .* sin.(theta ./ 2) )
end