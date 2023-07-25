function standard_config(; n::Int = 6)
    T = Float64
    W = T(3000e3)
    Omega = ComputationDomain(W, n)
    return Omega
end

function check_xy_ij()
    x, y = 1.0:10, 1.0:5
    X, Y = meshgrid(collect(x), collect(y))
    @test X[:, 1] == x
    @test Y[1, :] == y
end

function check_stereographic()
    lat, lon = -85.4, 67.8
    lat0, lon0 = -71.0, 0.0
    k, x, y = latlon2stereo(lat, lon, lat0, lon0)
    lat_, lon_ = stereo2latlon(x, y, lat0, lon0)
    @test lat_ ≈ lat
    @test lon_ ≈ lon

    latvec = -90.0:1.0:-60.0
    lonvec = -179.0:1.0:180.0
    Lon, Lat = meshgrid(lonvec, latvec)
    K, X, Y = latlon2stereo(Lat, Lon, lat0, lon0)
    Lat_, Lon_ = stereo2latlon(X, Y, lat0, lon0)
    @test Lat ≈ Lat_
    # @test Lon[2, :] ≈ Lon_[2, :]    # singularity at lat = -90°
end

function benchmark1()
    # Generating numerical results
    Omega = standard_config()
    c = PhysicalConstants()
    p = LateralVariability(Omega)

    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

    results = fastisostasy(t_out, Omega, c, p, Hcylinder, ODEsolver = BS3(),
        interactive_geostate = false, verbose = false)
    Omega, p = copystructs2cpu(Omega, p)

    # Comparing to analytical results
    slicex, slicey = slice_along_x(Omega)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    for k in eachindex(t_out)
        t = t_out[k]
        analytic_solution_r(r) = analytic_solution(r, t, c, p, H, R, analytic_support)
        u_analytic = analytic_solution_r.( get_r.(
            Omega.X[slicex, slicey], Omega.Y[slicex, slicey] .^ 2 ) )
        @test mean(abs.(results.viscous[k][slicex, slicey] .- u_analytic)) < 15
        @test maximum(abs.(results.viscous[k][slicex, slicey] .- u_analytic)) < 30
    end
end

function benchmark2()
    # Generating numerical results
    Omega = standard_config()
    c = PhysicalConstants(rho_ice = 0.931e3)
    G, nu = 0.50605e11, 0.28        # shear modulus (Pa) and Poisson ratio of lithsphere
    E = G * 2 * (1 + nu)
    lb = c.r_equator .- [6301e3, 5951e3, 5701e3]
    p = LateralVariability( Omega, layer_boundaries = lb,
        layer_viscosities = [1e21, 1e21, 2e21], litho_youngmodulus = E,
        litho_poissonratio = nu )
    t_out = years2seconds.([0.0, 1e3, 2e3, 5e3, 1e4, 1e5])
    sl0 = fill(-Inf, Omega.Nx, Omega.Ny)
    slicex, slicey = slice_along_x(Omega)
    x = Omega.X[slicex, slicey]

    for case in ["disc", "cap"]
        # Generate FastIsostasy results
        if occursin("disc", case)
            alpha = T(10)                       # max latitude (°) of uniform ice disc
            Hmax = T(1000)                      # uniform ice thickness (m)
            R = deg2rad(alpha) * c.r_equator    # disc radius (m), (Earth radius as in Spada)
            H_ice = stereo_ice_cylinder(Omega, R, Hmax)
        elseif occursin("cap", case)
            alpha = T(10)                       # max latitude (°) of ice cap
            Hmax = T(1500)
            H_ice = stereo_ice_cap(Omega, alpha, Hmax)
        end
        results = fastisostasy(t_out, Omega, c, p, H_ice, sealevel_0 = sl0,
            ODEsolver = BS3(), interactive_geostate = true, verbose = true)
        
        # Compare to 1D GIA models benchmark
        data = get_spada()
        for k in eachindex(t_out)
            u_itp = interpolate_spada_benchmark(c, data["u_$case"][k])
            dudt_itp = interpolate_spada_benchmark(c, data["dudt_$case"][k])
            n_itp = interpolate_spada_benchmark(c, data["n_$case"][k])

            m_u = mean(abs.(results.viscous[k][slicex, slicey] .- u_itp.(x)))
            m_dudt = mean(abs.(results.displacement_rate[k][slicex, slicey] .- dudt_itp.(x)))
            m_n = mean(abs.(results.geoid[k][slicex, slicey] .- n_itp.(x)))
            println("$m_u,  $m_dudt, $m_n")
        end
    end
end

function benchmark3()
    Omega = standard_config()
    c = PhysicalConstants()
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    t_out = years2seconds.( vcat(0:1_000:5_000, 10_000:5_000:50_000) )

    cases = ["gaussian_lo_D", "gaussian_hi_D", "gaussian_lo_η", "gaussian_hi_η",
        "no_litho", "ref"]
    seakon_files = ["E0L1V1", "E0L2V1", "E0L3V2", "E0L3V3", "E0L4V4", "E0L4V4"]
    for m in eachindex(cases)
        case = cases[m]
        file = seakon_files[m]
        p = choose_case(case, Omega, c)
        results = fastisostasy(t_out, Omega, c, p, Hcylinder, interactive_geostate = false,
            ODEsolver = BS3())
        idx = limit2slice()
        u_3DGIA = load_results("data/Latychev/$file", idx)
        itp = linear_interpolation(x, u_3DGIA[:, i], extrapolation_bc = Flat())

    end
end


function benchmark4()
end

function benchmark5()
end

function benchmark6()
end