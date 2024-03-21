
function comparison_figure(n)
    fig = Figure()
    axs = [Axis(fig[i, 1]) for i in 1:n]
    return fig, axs
end

function update_compfig!(axs::Vector{Axis}, fi::Vector, bm::Vector, clr)
    if length(axs) == length(fi) == length(bm)
        nothing
    else
        error("Vectors don't have matching length.")
    end

    for i in eachindex(axs)
        lines!(axs[i], bm[i], color = clr)
        lines!(axs[i], fi[i], color = clr, linestyle = :dash)
    end
end

function benchmark1_constants(Omega)
    c = PhysicalConstants(rho_litho = 0.0)
    p = LayeredEarth(Omega)
    t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
    
    εt = 1e-8
    pushfirst!(t_out, -εt)
    t_Hice = [-εt, 0.0, t_out[end]]

    R, H = 1000e3, 1e3
    Hcylinder = uniform_ice_cylinder(Omega, R, H)
    Hice = [zeros(Omega.Nx, Omega.Ny), Hcylinder, Hcylinder]

    return c, p, t_out, R, H, t_Hice, Hice
end


function benchmark1_compare(Omega, fip, H, R)
    # Comparing to analytical results
    ii, jj = slice_along_x(Omega)
    analytic_support = vcat(1.0e-14, 10 .^ (-10:0.05:-3), 1.0)
    fig, axs = comparison_figure(1)
    x, y = Omega.X[ii, jj], Omega.Y[ii, jj]
    cmap = cgrad(:jet, length(fip.out.t), categorical = true)
    for k in eachindex(fip.out.t)[2:end]
        t = fip.out.t[k]
        analytic_solution_r(r) = analytic_solution(r, t, fip.c, fip.p, H, R)
        u_analytic = analytic_solution_r.( get_r.(x, y) )
        update_compfig!(axs, [fip.out.u[k][ii, jj]], [u_analytic], cmap[k])
    end
    return fig
end


function slice_along_x(Omega::ComputationDomain)
    Nx, Ny = Omega.Nx, Omega.Ny
    return Nx÷2:Nx, Ny÷2
end

function benchmark1(dm)
    # Generating numerical results
    Omega = ComputationDomain(3000e3, 6, correct_distortion = false)
    c, p, t_out, R, H, t_Hice, Hice = benchmark1_constants(Omega)

    opts = SolverOptions(deformation_model = dm, verbose = true)

    fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts)
    solve!(fip)
    println("Computation took $(fip.out.computation_time) s")
    fig = benchmark1_compare(Omega, fip, H, R)
    # save("plots/benchmark1/plot-$dm.png", fig)
end

dm = :lv_elva
# benchmark1(dm)
Omega = ComputationDomain(3000e3, 6, correct_distortion = false)
c, p, t_out, R, H, t_Hice, Hice = benchmark1_constants(Omega)

opts = SolverOptions(deformation_model = dm, verbose = true)
fip = FastIsoProblem(Omega, c, p, t_out, t_Hice, Hice, opts = opts)

solve!(fip)

println("Computation took $(fip.out.computation_time) s")
fig = benchmark1_compare(Omega, fip, H, R)


u = copy(fip.now.u)
dudt = copy(fip.now.dudt)
t = 1.0
@btime update_diagnostics!(dudt, u, fip, t)

# Greatly reduced alloc for update_diagnostics! down to:
# without elastic update: 105.762 μs (12 allocations: 96.33 KiB)
# with elastic update: 1.125 ms (16 allocations: 128.39 KiB)