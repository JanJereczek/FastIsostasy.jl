#####################################################
# Unit conversion utils
#####################################################

global SECONDS_PER_YEAR = 60^2 * 24 * 365.25

"""
    years2seconds(t::Real)

Convert input time `t` from years to seconds.
"""
function years2seconds(t::T) where {T<:AbstractFloat}
    return t * T(SECONDS_PER_YEAR)
end

"""
    seconds2years(t::Real)

Convert input time `t` from seconds to years.
"""
function seconds2years(t::T) where {T<:AbstractFloat}
    return t / T(SECONDS_PER_YEAR)
end

"""
    m_per_sec2mm_per_yr(dudt::Real)

Convert displacement rate `dudt` from ``m \\, s^{-1} ``to ``mm \\, \\mathrm{yr}^{-1} ``.
"""
function m_per_sec2mm_per_yr(dudt::Real)
    return dudt * 1e3 * SECONDS_PER_YEAR
end

#####################################################
# Array utils
#####################################################

not(x::Bool) = !x
Base.fill(x::Real, fip::FastIsoProblem) = fill(x, fip.Omega)
Base.fill(x::Real, Omega::ComputationDomain) = Omega.arraykernel(fill(x, Omega.Nx, Omega.Ny))

function corner_matrix(T, Nx, Ny)
    M = zeros(T, Nx, Ny)
    M[1, 1], M[Nx, 1], M[1, Ny], M[Nx, Ny] = T.([1, 1, 1, 1])
    return M
end

"""
    samesize_conv_indices(N, M)

Get the start and end indices required for a [`samesize_conv`](@ref)
"""
function samesize_conv_indices(N, M)
    if iseven(N)
        j1 = M
    else
        j1 = M+1
    end
    j2 = 2*N-1-M
    return j1, j2
end

"""
    matrify(x, Nx, Ny)

Generate a vector of constant matrices from a vector of constants.
"""
function matrify(x::Vector{<:Real}, N::Int)
    return matrify(x, N, N)
end

function matrify(x::Vector{T}, Nx::Int, Ny::Int) where {T<:Real}
    X = zeros(T, Nx, Ny, length(x))
    @inbounds for i in eachindex(x)
        X[:, :, i] = fill(x[i], Nx, Ny)
    end
    return X
end

# function samesize_conv(X::M, Y::M, Omega::ComputationDomain{T, L, M}, bc::Function) where
#     {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
#     convo = bc(conv(X, Y), 2*Omega.Nx-1, 2*Omega.Ny-1)
#     return view(convo, Omega.i1:Omega.i2, Omega.j1:Omega.j2)
# end

# function samesize_conv(X::M, Y::M, Omega::ComputationDomain{T, L, M}) where
#     {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
#     convo = conv(X, Y)
#     # corner_bc!(convo, 2*Omega.Nx-1, 2*Omega.Ny-1, 0.0)
#     apply_bc!(convo, Omega.extended_bc_matrix, Omega.extended_nbc)
#     return view(convo, Omega.i1:Omega.i2, Omega.j1:Omega.j2)
# end

# Just a helper for blur! Not performant but we only blur at preprocessing
# so we do not care :)
function samesize_conv(X::M, Y::M, Omega::ComputationDomain) where
    {T<:AbstractFloat, M<:KernelMatrix{T}}
    (; i1, i2, j1, j2, convo_offset) = Omega
    ipc = InplaceConvolution(X, false)
    return samesize_conv(Y, ipc, i1, i2, j1, j2, convo_offset)
end
function samesize_conv(X, ipc, i1, i2, j1, j2, convo_offset)
    convo = ipc(X)
    return view(convo, i1+convo_offset:i2+convo_offset, j1-convo_offset:j2-convo_offset)
end

# Performant version for time loop
function samesize_conv(X::M, ipc::InplaceConvolution{T, C, FP, IP},
    Omega::ComputationDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T},
    M<:KernelMatrix{T}, C<:ComplexMatrix{T}, FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    convo = ipc(X)
    apply_bc!(convo, Omega.extended_bc_matrix, Omega.extended_nbc)
    return view(convo,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
end

function samesize_conv!(inout::M, convo_out::M, X::M,
    ipc::InplaceConvolution{T, C, FP, IP},
    Omega::ComputationDomain{T, L, M}) where {T<:AbstractFloat, L<:Matrix{T},
    M<:KernelMatrix{T}, C<:ComplexMatrix{T}, FP<:ForwardPlan{T}, IP<:InversePlan{T}}
    ipc(convo_out, X)
    inout .= view(convo_out,
        Omega.i1+Omega.convo_offset:Omega.i2+Omega.convo_offset,
        Omega.j1-Omega.convo_offset:Omega.j2-Omega.convo_offset)
    apply_bc!(inout, Omega.bc_matrix, Omega.nbc)
end

# function samesize_conv(X::CuMatrix{T}, Y::CuMatrix{T}, Omega::ComputationDomain{T, L, M}) where
#     {T<:AbstractFloat, L<:Matrix{T}, M<:KernelMatrix{T}}
#     convo = conv(X, Y)
#     corner_bc!(convo, 2*Omega.Nx-1, 2*Omega.Ny-1, 0.0)
#     return view(convo, Omega.i1:Omega.i2, Omega.j1:Omega.j2)
# end

"""
    write_out!(fip::FastIsoProblem)

Write results in output vectors if the load is updated internally.
If the load is updated externally, the user is responsible for writing results.
"""
function write_out!(fip::FastIsoProblem, k::Int)
    if fip.opts.internal_loadupdate
        fip.out.bsl[k] = fip.now.bsl
        fip.out.u[k] .= copy(Array(fip.now.u))
        fip.out.dudt[k] .= copy(Array(fip.now.dudt))
        fip.out.ue[k] .= copy(Array(fip.now.ue))
        fip.out.b[k] .= copy(Array(fip.now.b))
        fip.out.geoid[k] .= copy(Array(fip.now.geoid))
        fip.out.seasurfaceheight[k] .= copy(Array(fip.now.seasurfaceheight))
        fip.out.maskgrounded[k] .= copy(Array(fip.now.maskgrounded))
        fip.out.Hice[k] .= copy(Array(fip.now.H_ice))
        fip.out.Hwater[k] .= copy(Array(fip.now.H_water))
        fip.out.canomfull[k] .= copy(Array(fip.now.columnanoms.full))
        fip.out.canomload[k] .= copy(Array(fip.now.columnanoms.load))
        fip.out.canomlitho[k] .= copy(Array(fip.now.columnanoms.litho))
        fip.out.canommantle[k] .= copy(Array(fip.now.columnanoms.mantle))
    end
end

"""
    savefip(filename, fip; T = Float32)
Save the output of `fip::FastIsoProblem` as NetCDF file under `filename`.
"""
function savefip(filename, fip; T = Float32)

    ds = NCDataset(filename, "c")
    x, y, t = T.(fip.Omega.x), T.(fip.Omega.y), T.(fip.out.t)
    defDim(ds, "x", length(x))
    defDim(ds, "y", length(y))
    defDim(ds, "t", length(t))
    ncx = defVar(ds, "x", T, ("x",))
    ncy = defVar(ds, "y", T, ("y",))
    nct = defVar(ds, "t", T, ("t",))
    ncx[:] = x
    ncy[:] = y
    nct[:] = seconds2years.(t)

    append1D2nc!(ds, T, fip.out.bsl, "bsl")

    append2D2nc!(ds, T, log10.(fip.out.eta_eff), "log10 effective viscosity")
    append2D2nc!(ds, T, fip.out.maskactive, "active mask")

    append3D2nc!(ds, T, fip.out.u, "u")
    append3D2nc!(ds, T, fip.out.dudt, "dudt")
    append3D2nc!(ds, T, fip.out.ue, "ue")
    append3D2nc!(ds, T, fip.out.b, "b")
    append3D2nc!(ds, T, fip.out.geoid, "geoid")
    append3D2nc!(ds, T, fip.out.seasurfaceheight, "seasurfaceheight")
    append3D2nc!(ds, T, fip.out.maskgrounded, "maskgrounded")
    append3D2nc!(ds, T, fip.out.Hice, "Hice")
    append3D2nc!(ds, T, fip.out.Hwater, "Hwater")
    append3D2nc!(ds, T, fip.out.canomfull, "canomfull")
    append3D2nc!(ds, T, fip.out.canomload, "canomload")
    append3D2nc!(ds, T, fip.out.canomlitho, "canomlitho")
    append3D2nc!(ds, T, fip.out.canommantle, "canommantle")
    
    close(ds)

end

function append1D2nc!(ds, T, Z, var::String)
    ncZ = defVar(ds, var, T, ("t",))
    ncZ[:] = T.(Z)
    return nothing
end

function append2D2nc!(ds, T, Z, var::String)
    ncZ = defVar(ds, var, T, ("x", "y"))
    ncZ[:, :] = T.(Z)
    return nothing
end

function append3D2nc!(ds, T, Z, var::String)
    Z = cat(Z..., dims = 3)
    ncZ = defVar(ds, var, T, ("x", "y", "t"))
    ncZ[:, :, :] = T.(Z)
    return nothing
end

#####################################################
# Domain and projection utils
#####################################################
"""
    get_r(x::T, y::T) where {T<:Real}

Get euclidean distance of point (x, y) to origin.
"""
get_r(x::T, y::T) where {T<:Real} = sqrt(x^2 + y^2)

"""
    meshgrid(x, y)

Return a 2D meshgrid spanned by `x, y`.
"""
function meshgrid(x::V, y::V) where {T<:AbstractFloat, V<:AbstractVector{T}}
    one_x, one_y = ones(T, length(x)), ones(T, length(y))
    return x * one_y', one_x * y'
end

"""
    dist2angulardist(r::Real)

Convert Euclidean to angular distance along great circle.
"""
function dist2angulardist(r::T) where {T<:AbstractFloat}
    R = T(6371e3)       # radius at equator
    return 2 * atan( r / (2 * R) )
end

"""
    lon360tolon180(lon, X)

Convert longitude and field from `lon=0:360` to `lon=-180:180`.
"""
function lon360tolon180(lon, X)
    permidx = lon .> 180
    lon180 = vcat(lon[permidx] .- 360, lon[not.(permidx)])
    X180 = cat(X[permidx, :, :], X[not.(permidx), :, :], dims=1)
    return lon180, X180
end

"""
    scalefactor(lat::T, lon::T, lat0::T, lon0::T) where {T<:Real}
    scalefactor(lat::M, lon::M, lat0::T, lon0::T) where {T<:Real, M<:KernelMatrix{T}}

Compute scaling factor of stereographic projection for a given `(lat, lon)` and reference
`(lat0, lon0)`. Angles must be provided in radians.
Reference: [^Snyder1987], p. 157, eq. (21-4).
"""
function scalefactor(lat::T, lon::T, lat0::T, lon0::T; k0::T = T(1)) where {T<:Real}
    return 2*k0 / (1 + sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
end

function scalefactor(lat::M, lon::M, lat0::T, lon0::T; kwargs...,
    ) where {T<:Real, M<:KernelMatrix{T}}
    K = similar(lat)
    @inbounds for idx in CartesianIndices(lat)
        K[idx] = scalefactor(lat[idx], lon[idx], lat0, lon0; kwargs...)
    end
    return K
end

"""
    latlon2stereo(lat, lon, lat0, lon0)

Compute stereographic projection (x,y) for a given latitude `lat`
longitude `lon`, reference latitude `lat0` and reference longitude `lon0`.
Optionally one can provide `lat::KernelMatrix` and `lon::KernelMatrix`
if the projection is to be computed for the whole domain.
Note: angles must be provided in degrees!
Reference: John P. Snyder (1987), p. 157, eq. (21-2), (21-3), (21-4).
"""
function latlon2stereo(lat::T, lon::T, lat0::T, lon0::T;
    R::T = T(6371e3), kwargs...) where {T<:Real}
    lat, lon, lat0, lon0 = deg2rad.([lat, lon, lat0, lon0])
    k = scalefactor(lat, lon, lat0, lon0; kwargs...)
    x = R * k * cos(lat) * sin(lon - lon0)
    y = R * k * (cos(lat0) * sin(lat) - sin(lat0) * cos(lat) * cos(lon-lon0))
    return k, x, y
end

function latlon2stereo(lat::M, lon::M, lat0::T, lon0::T; kwargs...,
    ) where {T<:Real, M<:KernelMatrix{T}}
    K, X, Y = similar(lat), similar(lat), similar(lat)
    @inbounds for idx in CartesianIndices(lat)
        K[idx], X[idx], Y[idx] = latlon2stereo(lat[idx], lon[idx], lat0, lon0; kwargs...)
    end
    return K, X, Y
end

"""
    stereo2latlon(x, y, lat0, lon0)

Compute the inverse stereographic projection `(lat, lon)` based on Cartesian coordinates
`(x,y)` and for a given reference latitude `lat0` and reference longitude `lon0`.
Optionally one can provide `x::KernelMatrix` and `y::KernelMatrix`
if the projection is to be computed for the whole domain.
Note: angles must be  para elloprovided in degrees!

Convert stereographic (x,y)-coordinates to latitude-longitude.
Reference: John P. Snyder (1987), p. 159, eq. (20-14), (20-15), (20-18), (21-15).
"""
function stereo2latlon(x::T, y::T, lat0::T, lon0::T;
    R::T = T(6371e3), k0::T = T(1)) where {T<:Real}
    lat0, lon0 = deg2rad.([lat0, lon0])
    r = get_r(x, y) + 1e-8      # add small tolerance to avoid division by zero
    c = 2 * atan(r, 2*R*k0)
    lat = asin( cos(c) * sin(lat0) + y/r * sin(c) * cos(lat0) )
    lon = lon0 + atan( x*sin(c), ( - y * sin(lat0) * sin(c)) ) #(r * cos(lat0) * cos(c) - y * sin(lat0) * sin(c)) 
    return rad2deg(lat), rad2deg(lon)
end

function stereo2latlon(X::Matrix{T}, Y::Matrix{T}, lat0::T, lon0::T;
    kwargs...) where {T<:Real}
    Lat, Lon = copy(X), copy(X)
    @inbounds for idx in CartesianIndices(X)
        Lat[idx], Lon[idx] = stereo2latlon(X[idx], Y[idx], lat0, lon0; kwargs...)
    end
    return Lat, Lon
end


#####################################################
# Math utils
#####################################################

"""
    gauss_distr(X::KernelMatrix{T}, Y::KernelMatrix{T},
        mu::Vector{<:Real}, sigma::Matrix{<:Real})

Compute `Z = f(X,Y)` with `f` a Gaussian function parametrized by mean
`mu` and covariance `sigma`.
"""
function gauss_distr(X::M, Y::M, mu::Vector{T}, sigma::Matrix{T}) where
    {T<:AbstractFloat, M<:Matrix{T}}
    k = length(mu)
    G = similar(X)
    invsigma = inv(sigma)
    invsqrtdetsigma = 1/sqrt(det(sigma))
    @inbounds for i in axes(X,1), j in axes(X,2)
        G[i, j] = (2*π)^(-k/2) * invsqrtdetsigma * exp( 
            -0.5 * ([X[i,j], Y[i,j]] .- mu)' * invsigma * ([X[i,j], Y[i,j]] .- mu) )
    end
    return G
end

function generate_gaussian_field(
    Omega::ComputationDomain{T, M},
    z_background::T,
    xy_peak::Vector{T},
    z_peak::T,
    sigma::Matrix{T},
) where {T<:AbstractFloat, M<:Matrix{T}}
    if Omega.Nx == Omega.Ny
        N = Omega.Nx
    else
        error("Automated generation of Gaussian parameter fields only supported for" *
            "square domains.")
    end
    G = gauss_distr( Omega.X, Omega.Y, xy_peak, sigma )
    G = G ./ maximum(G) .* z_peak
    return fill(z_background, N, N) + G
end

function blur(X::AbstractMatrix, Omega::ComputationDomain, level::Real)
    if not(0 <= level <= 1)
        error("Blurring level must be a value between 0 and 1.")
    end
    T = eltype(X)
    sigma = diagm([(level * Omega.Wx)^2, (level * Omega.Wy)^2])
    kernel = T.(generate_gaussian_field(Omega, 0.0, [0.0, 0.0], 1.0, sigma))
    kernel ./= sum(kernel)
    # return copy(samesize_conv(Omega.arraykernel(kernel), Omega.arraykernel(X), Omega))
    return samesize_conv(kernel, X, Omega)
end


#####################################################
# Quadrature utils
#####################################################

"""
    get_quad_coeffs(T, n)

Return support points and associated coefficients with specified Type
for Gauss-Legendre quadrature.
"""
function get_quad_coeffs(T::Type, n::Int)
    x, w = gausslegendre(n)
    return T.(x), T.(w)
end


"""
    quadrature1D(f, n, x1, x2)

Compute 1D Gauss-Legendre quadrature of `f` between `x1` and `x2`
based on `n` support points.
"""
function quadrature1D(f::Union{Function, Interpolations.Extrapolation},
    n::Int, x1::T, x2::T) where {T<:AbstractFloat}
    x, w = get_quad_coeffs(T, n)
    m, p = get_normalized_lin_transform(x1, x2)
    sum = 0
    @inbounds for i=1:n
        sum = sum + f(normalized_lin_transform(x[i], m, p)) * w[i] / m
    end
    return sum
end

"""
    quadrature2D(f, x, w, x1, x2, y1, y2)

Return the integration of `f` over [`x1, x2`] x [`y1, y2`] with `x, w` some pre-computed
support points and coefficients of the Gauss-Legendre quadrature.
"""
function quadrature2D(
    f::Function,
    x::Vector{T},
    w::Vector{T},
    x1::T, x2::T,
    y1::T, y2::T,
) where {T<:AbstractFloat}

    n = length(x)
    mx, px = get_normalized_lin_transform(x1, x2)
    my, py = get_normalized_lin_transform(y1, y2)
    sum = T(0)
    @inbounds for i=1:n, j=1:n
        sum = sum + f(
            normalized_lin_transform(x[i], mx, px),
            normalized_lin_transform(x[j], my, py),
        ) * w[i] * w[j] / mx / my
    end
    return sum
end

"""
    get_normalized_lin_transform(x1, x2)

Return parameters of linear function mapping `x1, x2` onto `-1, 1`.
"""
function get_normalized_lin_transform(x1::T, x2::T) where {T<:AbstractFloat}
    x1_norm, x2_norm = T(-1), T(1)
    m = (x2_norm - x1_norm) / (x2 - x1)
    p = x1_norm - m * x1
    return m, p
end

"""
    normalized_lin_transform(y, m, p)

Apply normalized linear transformation with slope `m` and bias `p` on `y`.
"""
function normalized_lin_transform(y::T, m::T, p::T) where {T<:AbstractFloat}
    return (y-p)/m
end

#####################################################
# Kernel utils
#####################################################

null(Omega::ComputationDomain) = copy(Omega.null)

function kernelcollect(X, Omega)
    if not(Omega.use_cuda)
        return collect(X)
    else
        return X
    end
end

"""
    kernelpromote(X, arraykernel)

Promote X to the kernel (`Array` or `CuArray`) specified by `arraykernel`.
"""
function kernelpromote(X::M, arraykernel) where {M<:AbstractArray{T}} where {T<:Real}
    if isa(X, arraykernel)
        return X
    else
        return arraykernel(X)
    end
end

kernelpromote(X::Vector, arraykernel) = [arraykernel(x) for x in X]


"""
    reinit_structs_cpu(Omega, p)

Reinitialize `Omega::ComputationDomain` and `p::LayeredEarth` on the CPU, mostly
for post-processing purposes.
"""
function reinit_structs_cpu(Omega::ComputationDomain{T, M}, p::LayeredEarth{T, M}
    ) where {T<:AbstractFloat, M<:KernelMatrix{T}}

    Omega_cpu = ComputationDomain(Omega.Wx, Omega.Wy, Omega.Nx, Omega.Ny, use_cuda = false)
    p_cpu = LayeredEarth(
        Omega_cpu;
        layer_boundaries = Array(p.layer_boundaries),
        layer_viscosities = Array(p.layer_viscosities),
    )
    return Omega_cpu, p_cpu
end

function choose_fft_plans(X, use_cuda)
    if use_cuda
        pfft! = CUFFT.plan_fft!(complex.(X))
        pifft! = CUFFT.plan_ifft!(complex.(X))
    else
        pfft! = plan_fft!(complex.(X))
        pifft! = plan_ifft!(complex.(X))
    end
    return pfft!, pifft!
end

# function remake!(fip::FastIsoProblem)
#     @set fip.now = CurrentState(fip.Omega, fip.ref)
#     println(extrema(fip.now.u))
#     if fip.opts.dense_output
#         @set fip.out = DenseOutputs(fip.Omega, fip.out.t,
#             fip.p.effective_viscosity, fip.ref.maskactive)
#     else
#         @set fip.out = SparseOutputs(fip.Omega, fip.out.t)
#     end
#     return nothing
# end

function remake!(fip::FastIsoProblem)
    # Get values from ReferenceState
    fip.now.u .= copy(fip.ref.u)
    fip.now.ue .= copy(fip.ref.ue)
    fip.now.seasurfaceheight .= copy(fip.ref.seasurfaceheight)
    fip.now.H_water .= copy(fip.ref.H_water)
    fip.now.H_ice .= fip.tools.Hice(fip.out.t[1])
    fip.now.b .= copy(fip.ref.b)

    # Some values are not included in ReferenceState and need to be init with 0.
    fip.now.dudt .= null(fip.Omega)
    fip.now.geoid .= null(fip.Omega)
    fip.now.countupdates = 0
    fip.now.columnanoms = ColumnAnomalies(fip.Omega)

    out = DenseOutputs(fip.Omega, fip.out.t,
        fip.p.effective_viscosity, fip.ref.maskactive)
    fip.out.u = out.u
    fip.out.ue = out.ue
    return nothing
end

#####################################################
# BC utils
#####################################################

function periodic_extension(M::Matrix{T}, Nx::Int, Ny::Int) where {T<:AbstractFloat}
    M_periodic = zeropad_extension(M, Nx, Ny)
    M_periodic[1, 2:end-1] .= M[end, :]
    M_periodic[end, 2:end-1] .= M[1, :]
    M_periodic[2:end-1, 1] .= M[:, end]
    M_periodic[2:end-1, end] .= M[:, 1]
    return M_periodic
end

function zeropad_extension(M::Matrix{T}, Nx::Int, Ny::Int) where {T<:AbstractFloat}
    M_zeropadded = fill(T(0), Nx+2, Ny+2)
    M_zeropadded[2:end-1, 2:end-1] .= M
    return M_zeropadded
end

function init()
    println("Initializing CUDA Stencil")
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.allowscalar(false)
end

#####################################################
# Example utils
#####################################################

function mask_disc(X::KernelMatrix{T}, Y::KernelMatrix{T}, R::T;
    center::Vector{<:Real}) where {T<:AbstractFloat}
    return mask_disc(sqrt.((X .- center[1]) .^ 2 + (Y .- center[2]) .^ 2), R)
end

function mask_disc(r::KernelMatrix{T}, R::T) where {T<:AbstractFloat}
    return T.(r .< R)
end

function uniform_ice_cylinder(Omega::ComputationDomain, R::T, H::T;
    center::Vector{<:Real} = [0.0, 0.0]) where {T<:AbstractFloat}
    M = mask_disc(Omega.X, Omega.Y, R, center = center)
    return M .* H
end

function stereo_ice_cylinder(
    Omega::ComputationDomain,
    R::T,
    H::T,
) where {T<:AbstractFloat}
    M = mask_disc(Omega.R, R)
    return M .* H
end

function stereo_ice_cap(
    Omega::ComputationDomain,
    alpha_deg::T,
    H::T,
) where {T<:AbstractFloat}
    alpha = deg2rad(alpha_deg)
    M = Omega.Theta .< alpha
    return H .* sqrt.( M .* (cos.(Omega.Theta) .- cos(alpha)) ./ (1 - cos(alpha)) )
end