using FastIsostasy, CairoMakie, FFTW

function make_empty_ax()
    fig = Figure()
    ax = Axis(fig[1,1], aspect = DataAspect())
    hidedecorations!(ax)
    return fig, ax
end

n = 7
Omega = ComputationDomain(3000e3, n, projection_correction = false)
c = PhysicalConstants()
p = LayeredEarth(Omega)
alpha = 10.0                        # max latitude (°) of ice cap
Hmax = 1500.0
Hice = stereo_ice_cap(Omega, alpha, Hmax)
fig, ax = make_empty_ax()
heatmap!(ax, Hice, colormap = :greys)
# save("src/logo/ice_cap.png", fig)


t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])
fip = FastIsoProblem(Omega, c, p, t_out, false, Hice, verbose = true)
solve!(fip)
println("Computation took $(fip.out.computation_time) s")
fig, ax = make_empty_ax()
heatmap!(ax, fip.out.u[end] + fip.out.ue[end], colormap = :greys)
# save("src/logo/displacement.png", fig)

fig, ax = make_empty_ax()
heatmap!(ax, Hice + fip.out.u[end] + fip.out.ue[end], colormap = :greys)
# save("src/logo/sum.png", fig)

slice = fip.out.u[end][:, Omega.My]
messy_fft_slice = fft(slice)
fft_slice = vcat(messy_fft_slice[Omega.My+1:end], messy_fft_slice[1:Omega.My])
fig = Figure(resolution = (3200, 900))
ax = Axis(fig[1,1])
hidedecorations!(ax)
lines!(ax, abs.(fft_slice), linewidth = 30, color=("#CB3C33", 1.0))
fig
save("src/logo/fft.png", fig)
