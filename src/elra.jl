"""
    get_kei(filt, L_w, dx, dy; T = Float64)

Calculate the Kelvin filter in 2D.

# Arguments
- `filt::Matrix`: The filter array to be filled with Kelvin function values.
- `L_w::Real`: The characteristic length scale.
- `dx::Real`: The grid spacing in the x-direction.
- `dy::Real`: The grid spacing in the y-direction.

# Returns
- `filt::Matrix{T}`: The filter array filled with Kelvin function values.
"""
function get_kei(Omega::ComputationDomain{T, <:Any, <:Any}, L_w) where
    {T<:AbstractFloat}

    (;nx, ny, dx, dy) = Omega
    if nx != ny
        error("The Kelvin filter is only implemented for square domains.")
    end
    n2 = (nx-1) ÷ 2

    # Load tabulated values and init filt before filling with loop
    rn_vals, kei_vals = load_viscous_kelvin_function(T)
    filt = null(Omega)
    for j = -n2:n2
        for i = -n2:n2
            x = i*dx
            y = j*dy
            r = sqrt(x^2 + y^2)

            # Get actual index of array
            i1 = i + 1 + n2
            j1 = j + 1 + n2

            # Get correct kei value for this point
            filt[i1, j1] = get_kei_value(r, L_w, rn_vals, kei_vals)
        end
    end

    return filt
end

"""
    get_kei_value(r, L_w, rn_vals, kei_vals)

Calculate the Kelvin function (kei) value based on the radius from the point load `r`,
the flexural length scale `L_w`, and the arrays of normalized radii `rn_vals` and
corresponding kei values `kei_vals`.

This function first normalizes the radius `r` by the flexural length scale `L_w` to get
the current normalized radius. If this value is greater than the maximum value in `rn_vals`,
the function returns the maximum value in `kei_vals`. Otherwise, it finds the interval in
`rn_vals` that contains the current normalized radius and performs a linear interpolation
to calculate the corresponding kei value.
"""
function get_kei_value(r, L_w, rn_vals, kei_vals)
    n = length(rn_vals)

    # Get current normalized radius from point load
    rn_now = r / L_w

    if rn_now > rn_vals[n]
        kei = kei_vals[n]
    else
        k = 1
        while k < n
            if rn_now >= rn_vals[k] && rn_now < rn_vals[k+1]
                break
            end
            k += 1
        end

        # Linear interpolation to get current kei value
        kei = kei_vals[k] + (rn_now - rn_vals[k]) / (rn_vals[k+1] - rn_vals[k]) *
            (kei_vals[k+1] - kei_vals[k])
    end

    return kei
end

"""
    get_flexural_lengthscale(litho_rigidity, rho_uppermantle, g)

Compute the flexural length scale, based on Coulon et al. (2021), Eq. in text after Eq. 3.
The flexural length scale will be on the order of 100km.

# Arguments
- `litho_rigidity`: Lithospheric rigidity
- `rho_uppermantle`: Density of the upper mantle
- `g`: Gravitational acceleration

# Returns
- `L_w`: The calculated flexural length scale
"""
function get_flexural_lengthscale(litho_rigidity, rho_uppermantle, g)
    L_w = (litho_rigidity / (rho_uppermantle*g)) .^ 0.25
    return L_w
end

"""
    calc_viscous_green(GV, kei2D, L_w, D_lith, dx, dy)

Calculate the viscous Green's function. Note that L_w contains information about
the density of the upper mantle.

"""
function calc_viscous_green(Omega, litho_rigidity, kei, L_w)
    return -L_w^2 ./ (2*pi*litho_rigidity) .* kei .* (Omega.dx*Omega.dy)
end


# E = 66.0
# He_lith = 88.0
# nu = 0.28
# D_lith = (E*1e9) * (He_lith*1e3)^3 / (12.0 * (1.0-nu^2))