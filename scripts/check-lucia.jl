using FastIsostasy, NCDatasets, Interpolations

function main()
    # Load the ice history from Yelmo
    ds = NCDataset("scripts/LGM_equilibrium_15kyr.nc", "r")
    x = copy(ds["xc"][:])
    y = copy(ds["yc"][:])
    t = copy(ds["time"][:])
    Hice = copy(ds["H_ice"][:, :, :])
    zbed = copy(ds["z_bed"][:, :, :])
    close(ds)

    # Define the FastIsostasy domain, which is alpha times bigger than in Yelmo
    xiso = Float64.( x .- mean(x) ) * 1e3
    yiso = Float64.( y .- mean(y) ) * 1e3
    tiso = Float64.(reverse(-t))
    Hice_ordered = Float64.(reverse(Hice, dims = 3))
    nx, ny, nt = length(xiso), length(yiso), length(tiso)
    alpha = 1.5
    Omega = ComputationDomain(maximum(xiso) * alpha, maximum(yiso) * alpha, nx, ny)
    c = PhysicalConstants()
    p = LayeredEarth(Omega)

    # Compute the ice thickness anomaly
    Hice_itp = linear_interpolation((xiso, yiso, tiso), Hice_ordered, extrapolation_bc = 0.0)
    Hice_iso = ([Hice_itp.(Omega.X, Omega.Y, tt) for tt in tiso])
    dHice = [Hice_iso_snap - Hice_iso[1] for Hice_iso_snap in Hice_iso]

    # Define the problem and solve it
    fip = FastIsoProblem(Omega, c, p, years2seconds.(tiso), false,
        years2seconds.(tiso), dHice)
    solve!(fip)
    println("Computation took $(fip.out.computation_time)")

    # Save results as .nc
    ds = NCDataset("/home/jan/fastiso-results-alpha=$alpha.nc", "c")
    defDim(ds, "x", nx)
    defDim(ds, "y", ny)
    defDim(ds, "t", nt)
    ds.attrib["title"] = "FastIsostasy.jl results on Greenland deglaciation"

    xx = defVar(ds, "x", Float64, ("x",))
    xx[:] = round.(Omega.x)
    yy = defVar(ds, "y", Float64, ("y",))
    yy[:] = round.(Omega.y)
    tt = defVar(ds, "t", Float64, ("t",))
    tt[:] = t

    v = defVar(ds, "u", Float64, ("x", "y", "t"))
    v[:,:,:] = cat([fip.out.u[k] + fip.out.ue[k] for k in 1:nt]..., dims = 3)
    v.attrib["units"] = "meters"

    w = defVar(ds, "dHice", Float64, ("x", "y", "t"))
    w[:,:,:] = cat([dH for dH in dHice]..., dims = 3)
    w.attrib["units"] = "meters"

    close(ds)
end

main()