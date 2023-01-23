push!(LOAD_PATH, "../")
using FastIsostasy
using JLD2
include("helpers_compute.jl")

@inline function main(
    n::Int,                     # 2^n x 2^n cells on domain, (1)
    case::String;               # Application case
    use_cuda = true::Bool,
)

    if use_cuda
        kernel = "gpu"
    else
        kernel = "cpu"
    end

    T = Float64
    L = T(3000e3)               # half-length of the square domain (m)
    Omega = init_domain(L, n, use_cuda = use_cuda)
    R = T(1000e3)               # ice disc radius (m)
    H = T(1000)                 # ice disc thickness (m)
    filename = "$(case)_$(kernel)_N$(Omega.N)"
    println("Computing $case on $(Omega.N) x $(Omega.N) grid...")

    c = init_physical_constants()
    if case == "binaryD"
        binary_thickness = generate_binary_field(Omega, 50e3, 250e3)
        binary_rigidity = get_rigidity.(binary_thickness)
        p = init_multilayer_earth(Omega, c, litho_rigidity = binary_rigidity)
    elseif case == "binaryη"
        binary_viscosity = generate_binary_field(Omega, 1e18, 1e23) # + rand(Omega.N, Omega.N)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N) # + 1e15 .* rand(Omega.N, Omega.N)
        layers_viscosity = cat(binary_viscosity, halfspace_viscosity, dims=3)
        layers_begin = matrify_vectorconstant([88e3, 400e3], Omega.N)
        p = init_multilayer_earth(
            Omega,
            c,
            layers_begin = layers_begin,
            layers_viscosity = layers_viscosity,
        )
    elseif case == "binaryDη"
        binary_thickness = generate_binary_field(Omega, 50e3, 250e3)
        binary_rigidity = get_rigidity.(binary_thickness)
        binary_viscosity = generate_binary_field(Omega, 1e18, 1e23)
        halfspace_viscosity = fill(1e21, Omega.N, Omega.N)
        layers_viscosity = cat(binary_viscosity, halfspace_viscosity, dims=3)
        p = init_multilayer_earth(
            Omega,
            c,
            litho_rigidity = binary_rigidity,
            layers_viscosity = layers_viscosity,
        )
    end

    t_out_yr = [0.0, 1.0, 1e1, 1e2, 1e3, 2e3, 5e3, 1e4, 1e5]
    t_out = years2seconds.(t_out_yr)

    u3D = zeros( T, (size(Omega.X)..., length(t_out)) )
    u3D_elastic = copy(u3D)
    u3D_viscous = copy(u3D)
    dudt3D_viscous = copy(u3D)

    sigma_zz_disc = generate_uniform_disc_load(Omega, c, R, H)
    placeholder = 1.0
    tools = precompute_terms(placeholder, Omega, p, c)

    t1 = time()
    @time forward_isostasy!(
        Omega,
        t_out,
        u3D_elastic,
        u3D_viscous,
        dudt3D_viscous,
        sigma_zz_disc,
        tools,
        p,
        c,
    )
    t_fastiso = time() - t1

    if use_cuda
        Omega, p = copystructs2cpu(Omega, p, c)
    end

    jldsave(
        "data/test3/$filename.jld2",
        u3D_elastic = u3D_elastic,
        u3D_viscous = u3D_viscous,
        dudt3D_viscous = dudt3D_viscous,
        sigma_zz = sigma_zz_disc,
        Omega = Omega,
        c = c,
        p = p,
        R = R,
        H = H,
        t_fastiso = t_fastiso,
        t_out = t_out,
    )
end

#= Application cases:
    - "binaryD"
    - "binaryη"
    - "binaryDη"
=#
for n in 4:6
    for case in ["binaryD", "binaryη", "binaryDη"]
        main(n, case, use_cuda = false)
    end
end