push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

# function main()
#     n::Int,                     # 2^n x 2^n cells on domain, (1)
#     case::String;               # Application case
#     use_cuda = false::Bool,
# )
n = 7
case = "refactor"
T = Float64
L = T(3000e3)               # half-length of the square domain (m)
Omega = init_domain(L, n, use_cuda = false)
c = init_physical_constants()
p = init_multilayer_earth(Omega, c)

R = T(1000e3)               # ice disc radius (m)
H = T(1000)                 # ice disc thickness (m)

filename = "$(case)_N$(Omega.N)"
println("Computing $case on $(Omega.N) x $(Omega.N) grid...")

t_out = years2seconds.([0.0, 100.0, 500.0, 1500.0, 5000.0, 10_000.0, 50_000.0])

Hice = uniform_ice_cylinder(Omega, R, H)
t_Hice_snapshots = [t_out[1], t_out[end]]
Hice_snapshots = [Hice, Hice]

t_eta_snapshots = [t_out[1], t_out[end]]
eta_snapshots = [p.effective_viscosity, p.effective_viscosity]

tools = precompute_fastiso(Omega, p, c)
t1 = time()
results = isostasy(t_out, Omega, tools, p, c,
    t_Hice_snapshots, Hice_snapshots, t_eta_snapshots, eta_snapshots)
t_fastiso = time() - t1
println(t_fastiso)

jldsave(
    "data/test1/$filename.jld2",
    Omega = Omega,
    c = c,
    p = p,
    results = results,
    t_fastiso = t_fastiso,
)

# """
# Application cases:
#     - "cn2layers"
#     - "cn3layers"
#     - "euler2layers"
#     - "euler3layers"
# """
# case = "euler3layers"
# for n in 4:5
#     main(n, case, use_cuda = true)
# end