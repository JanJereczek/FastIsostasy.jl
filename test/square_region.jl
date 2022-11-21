using FastIsostasy

T = Float64

p = init_solidearth_params()
c = init_physical_constants()

timespan = T.([0, 1e3]) * T(c.seconds_per_year)     # (s)
dt = T(1) * T(c.seconds_per_year)                   # (s)

L = T(2e6)              # half-length of the square domain (m)
N = 2^10                # number of cells on domain (1)
d = init_domain(L, N)   # domain parameters

u2D = zeros(size(d.X))  # init zero field of displacement (m)
sigma_zz = copy(u2D)    # first, test a zero-load field (N/m^2)
tools = init_integrator_tools(dt, u2D, d, p, c)
u_next = forwardstep_isostasy(dt, u2D, sigma_zz, tools)